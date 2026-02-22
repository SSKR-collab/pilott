/*
 * Distributed RockNet - Linear Classifier with QADAM Optimizer
 * Adapted from cp_firmware/app/linear_classifier.c for POSIX simulation
 *
 * Each node has its own subset of features & weights.
 * Partial logits are computed per-node, then aggregated in main.c via shared memory.
 * Gradient computation uses the AGGREGATED logits (sum of all nodes' partial logits).
 * QADAM (quantized Adam) optimizer for memory-efficient training.
 */
#include "linear_classifier.h"
#include "rocket_config.h"
#include "conv.h"
#include "dynamic_tree_quantization.h"

#include <math.h>
#include <stdio.h>

#define USE_QADAM 1

extern uint16_t TOS_NODE_ID;

static float weight[NUM_CLASSES * MAX_FEATURES_PER_DEVICE];
static float bias[NUM_CLASSES] = {0};

static float d_bias[NUM_CLASSES] = {0};
static float d_weight[NUM_CLASSES * MAX_FEATURES_PER_DEVICE] = {0};
static float features[MAX_FEATURES_PER_DEVICE];

static uint32_t num_d = 0;

/* Fixed random numbers for weight initialization (same as firmware) */
static const float random_numbers[] = {-0.16050057663875206, 0.5970915755927826, 2.061492107386712, -0.2154174139690511, 0.626780507165539, 0.09408583379568887, 0.4389517701364244, 0.7698918830352949, 0.43177410664674987, -1.1574003112112194, -0.7004444411346599, 0.09298660903631278, 1.1059020921144638, -0.1870404079433893, 1.892048166441975, -1.541315417399921, 0.3221545260816847, 1.0108077384165326, 1.2456964954648364, 0.12002961216766801, -0.8274810322980681, 1.1732581168764415, 0.7439977517156922, -1.2290736933666262, -0.6593800945013918, 0.16606587543412055, 0.7318562731305878, -2.2810358994659645, 0.31765283175535286, -1.4482213048186283, 0.4104443374541629, -0.7190956603596697, -0.7629588702128622, -0.4464195757710106, -0.9306685036788358, 0.395848785769842, 0.226991733147036, -0.0943502862000136, -0.9215569463302801, 0.05775763306062417, -0.2633378810885629, -1.0970541708273354, -1.8035791457492107, 0.20580980143508146, -0.3036396293223726, -1.6782141135427622, 1.5067663257982702, -0.681425296035707, 0.5746223266362493, -1.5157575245689359, -0.213174613550889, -0.5322989598456239, 0.3420867337874028, -1.586458729870498, -1.00732775215969, 2.6776430878072457, 0.7183019389784476, -0.08267752725515579, -0.5136777999396469, -0.12461675444218326, -0.43876800245468334, -1.5891937011633492, 0.36875001163472476, -1.8259143779109195, 0.31202401846964106, 0.19991660262228347, 0.354441995986561, 1.4844332513861085, -0.2773430013243048, -1.2258712832322496, 2.618032466466109, -2.1572991172940177, -1.239549588480878, 0.13805232344174678, -1.2757388656389528, 0.6677428603437062, -0.35265098238516907, -1.0865715416434814, -1.2685593430048148, 0.010079259750744689, 1.33428865068095, 0.2447157330793637, 0.36581529976964605, -0.4831290140173886, 1.2749844694397912, -1.0615852801713037, 1.0019206088330903, 0.4953796270336336, 0.18677188020525998, 1.1756744421260696, -0.016844209525477466, 2.0589218344041647, -1.3842586869810356, 0.037915446698647645, 0.2759979457565606, 0.3176591679724005, -0.22420012486494179, 0.11280907288754739, 2.4502890657710807, -1.8224095093365686, 0.7889570134725457, 0.11462360173147039, -0.5327819035928176, 0.18420098052334996, -0.08233228678952378, -1.9381916659759921, -1.710513610754408, -0.8307032093426779, -0.7309430948297347, -0.01958669173381063, -0.4095012909984872, 0.25877875970486575, 0.8380850844330884, 0.17467293067330866, 1.3734402365962672, 1.1960777560393954, 0.2681002567028299, 1.5523741496678294, -0.8774700140571056, -0.00040230678252818955, 0.30636536819318816, -1.0791552310973758, 1.5448227762273148, -0.5726451184133842, 0.6647110070754098, 0.21560858593092844, -1.114300608048537, 0.7146263971807912, 0.4877236761090521, -2.8174402747398277, 1.0792128321762582, 0.5177384688169494, -0.6929855319369413, -0.29453339082741475, 1.0635583029193136, 0.4744807374517816, -1.1967523571828513, -0.6109074107340392, -1.528466147342271, -0.13547452339629004, 1.358972273674476, -0.7541893819831831, -0.7705316793207387, -0.07056018935256035, 1.0682284126727113, -1.5999538757235636, 0.7246030717711746, 0.3966292480786254, 0.5869280843511319, 0.40248373661198444, -0.22036174173849044, 0.867184310621377, -2.044112967395328, -1.11734082975259, -0.528390933399204, -0.4123074736954984, 0.5730025296970032, -0.6859837565060568, -0.5472943228265468, -1.064603946697456, 1.5787075896208587, 0.9032845082949005, -1.4708220791958855, -0.7137338138331746, -0.13873827969670346, 0.31518652213294107, -1.6226454796815677, 1.8180361600467638, -0.9475384703590347, -1.798448041364569, 0.5454411790208222, 0.3924221695487607, -1.861358501991749, 0.14051606507119532, 0.851104100882436, 0.8919386815744247, 0.7744285554546163, -0.4319564432646162, 1.1226101388017782, -0.6039736266208416, -1.68199951282669, -0.1802245143435506, -0.6498131081996948, -1.4877776314653475, -0.8582637451066997, 1.1336013255426571, 1.1680528033938506, -0.01460991848678038, -0.19760826327130768, -0.2881632143001408, -0.9505451746195609, -0.3760872095981079, -0.5611992921962772, 1.2532725613126683, 0.11482462361900501, -0.22457239911392077, -0.16279272447090823, -2.184285006642186, 0.37604943268687063, 1.817044910710989, 0.8295209243595448, -0.8209496113598641, 0.6602501813489228, -0.10895526132793376, -0.8385657373171268, -0.04290400147532712, -1.1275927878272254, -0.25217693322519846, 1.9416299931279932, 0.6727245293222568, -0.14456734272934726, 0.39849286968321757, 1.071434039716485, 0.024022222833383237, 0.7562204907626596, -1.2700653159866089, -0.4490189971892423, -1.0216485806346196, 1.2916528796599835, 0.6538564385373978, -0.19541089753349913, 1.3253284298600656, -0.804776397183831, 0.5031230985321449, 0.953189431154473, 0.3012273625115138, -0.1818510027814175, 0.9880129628267009, 0.7223829624761856, -0.4692810637742843, -0.31683202192964843, -0.1884577088164422, 0.7407130549295763, -2.82555038063954, -0.07236828841590548, -0.7256464536081229, 0.09101811455455672, -1.8561363636925874, 0.4530034421798358, -0.09167138047895573, -1.2984545363989661, 0.10667343346172085, -1.4184433855309182, -1.307220204707491, 0.4970302978110486, -0.10594934742087973, 0.6950876873648514, -0.14748570995557803, 0.49452701905690083, 1.0124705115856507, 0.03625603607445004, -1.7106393178982833, 0.9826004190841309, 1.590654306099898, -0.31260818958383463, -0.002080525880003823, -1.7207349975349515, -0.21464580879288048, 0.41031880646611724, 1.8766880169058384, 0.3675836967533125, 1.397433961327472, 0.8515265305976116, 0.5039805150839376, 0.6310756342060004, 0.04117244316509129, 1.096269737377604, -0.8203797610157714, -0.8276351439013999, 0.03610177179794255, 0.9203358493124055, -1.4716748174651275, -0.5954592430052775, 0.20487496997336402, -0.6044236616910987, -0.14117042193855392, -1.3549877821302354, 0.049799481236227724, -0.5318539146451227, 1.5359705436324607, 1.0773901770793455, -1.3709893765916803, 1.3120978820240097, 0.8657802820416781, -0.25199593953318694, 0.8372063690338459, 1.1981606894218033, -0.09766842957446965, -0.3506468831228084, 0.45287456659127756, -1.2534296226688815, 1.775325122972415, 1.1739672642695524, 2.219040278196035, 0.9596430575672066, -0.6621441163276752, -1.7264757628559046, 0.10792181257546503, 1.3568535770175538, 1.225161064890428, -1.8395881518250317, -1.8298887253031026, 0.49835809259447, -1.062705635123759, -0.8720113971356597, 1.1881245906148596, 0.03963948960480389, -0.680065725722833, 0.5695181018145987, -0.4311285297904952, 1.2524590254623431, 1.3515832767663871, -0.6324709070502756, -0.08753764791866314, -1.4040323656675702, 0.2238555288633269, 0.45263215722166456, 0.7224953852902898, -0.026367431068511717, 0.7686630468664448, -0.21327690241002448, -1.059609162406975, -0.14386083368873556, 1.1287796561957724, -0.38908978643662945, -0.7551550382339312, 1.9171437934009887, -0.2898094952722943, -1.38661743430188, 0.9447811183594097, -0.8791815397453276, -0.3613721069054875, 2.447220243007632, 0.3365632706272984, 0.43739639818409476, -0.715426073713043, 1.1427166163358917, 0.5634820979929855, 1.4740941166099757, 0.7411766521279626, 2.119328552472476, 0.5219097767956409, -0.26742928902356256, -2.0098201281315298, -0.06336347388175793, -0.40334448399739553, -0.1893420749404619, 0.42952288467115923, 0.9788893793390452, 1.594312080223174, -0.5054821272483843, -0.446948238367112, 0.347135575154355, -1.8625023745306026, -0.20902381448196666, -2.1200885295355367, 0.2827363529899556, 0.34613334498984955, -1.3137302632921732, 0.19236165465607497, 0.9851544728458768, 0.6985791477351603, 1.263036413394137, -0.2317870583000212, 0.6800387034148027, -0.17263629531527633, -0.014149556722172574, -0.6544177065398544, 0.023102559759668573, 1.4462657987070089, -0.056874392742433645, 1.3300128871840402, 0.45875061930166816, 0.6358725715705493, -1.0208048735536963, -0.19713389833365735, -0.8243250134061837, -0.04964388753161731, -0.4121443038093397, 1.0468066187941354, 0.33411459220023676, 0.40072280893013273, 1.0709273268203683, 0.3443325532139713, -0.13744595302615176, 0.11010002858215345, 0.9512438318980155, -0.22038080115935457, -0.033700028724228163, 0.5957724807949181, 0.7366918902719435, -0.5069773566245456, -0.9579884809930904, -0.15499217940628862, -1.8286177835720248, 0.7467359054766698, 0.274697702964015, -0.7956936028827747, 0.8707444205741989, 1.7130959223551083, 0.42285388364170023, -0.08776026355522577, 0.023989038779832147, -1.435736813493663, 0.7931059026645249, -0.5401330120586229, -0.8663491865096954, -1.812743200775639, -0.711998784801694, 0.3944158130697042, -0.7150343668592207, 0.8424323037005439, -1.5426794020417989, -0.004457979761406293, 0.007995016419002559, -0.2703149670247228, -1.2771751604177166, -0.43448543228842307, -1.0029995006985084, -0.13055734410307937, 0.5231926906468535, -0.9946676797178857, -1.0776012970201319, 0.7926724765151568, 1.0747153897871136, -0.5480906153967463, 0.22620177935911337, 1.360387532611909, 1.7181567185200064, -0.6604823182499624, -0.19896777860274895, -0.1534648648619949, 2.814494923893279, -0.5839513403254775, -0.4092543738776122, -0.7133301377742531, -0.6914432848641713, 0.16103652069190447, 1.332571287971133, -0.8690109499562922, -0.12772990020642866, -0.7706920890250923, -1.2625897780184205, -1.8154715849661864, 1.5274619695609177, 1.2738513434175425, 0.8256620435567037, -0.29561314291705554, -0.16666545717721, -0.09019468440691422, -0.4365077184193798, -0.6967997376077948, -1.551126865747258, 0.4463420898455036, -1.3859777139021994, 0.7745722907025537, -0.22154311261416726, -0.5057344504484459, -0.6500955158509402, 0.2513099179906473, 0.6225585885785163, 2.7436859218250746, -0.6201362265455349, -1.64373181141702, 2.995501263092536, 0.14622647988502402, 1.5648368078308783, -0.7842926913801277, 0.386106737049989, 0.7683051234748585, 1.4155086912409625, -1.2244347411816792, 1.5323002257053802, -0.26608178466308063, 0.18060686936395007, 0.826058862767561, -0.4362506783380156, -0.010024336782672463, 1.1452204251761533, -1.4401741911764943, 1.0902711951475297, 1.4747804806465097, -1.2083477512858738};

#define WEIGHT_DECAY (0)

#if USE_QADAM
typedef uint8_t adam_dtype_t;
#else
typedef float adam_dtype_t;
#endif

static adam_dtype_t m_t_bias[NUM_CLASSES] = {0};
static adam_dtype_t v_t_bias[NUM_CLASSES] = {0};
static adam_dtype_t m_t_weight[NUM_CLASSES * MAX_FEATURES_PER_DEVICE] = {0};
static adam_dtype_t v_t_weight[NUM_CLASSES * MAX_FEATURES_PER_DEVICE] = {0};

#if USE_QADAM
#define QADAM_BUFFER_SIZE (256)
#define NUM_BIAS_SCALINGS (NUM_CLASSES / QADAM_BUFFER_SIZE + 1)
#define NUM_WEIGHT_SCALINGS (NUM_CLASSES * MAX_FEATURES_PER_DEVICE / QADAM_BUFFER_SIZE + 1)
static float m_t_bias_scalings[NUM_BIAS_SCALINGS] = {0};
static float v_t_bias_scalings[NUM_BIAS_SCALINGS] = {0};
static float m_t_weight_scalings[NUM_WEIGHT_SCALINGS] = {0};
static float v_t_weight_scalings[NUM_WEIGHT_SCALINGS] = {0};

static float block_wise_quantization_m_buffer[QADAM_BUFFER_SIZE];
static float block_wise_quantization_v_buffer[QADAM_BUFFER_SIZE];
#endif

#define BETA_1 (0.9f)
#define BETA_2 (0.999f)
#define EPSILON (1e-8)
#define LEARNING_RATE (1e-3)

static float out_softmax[NUM_CLASSES];

static uint32_t t = 1;
uint8_t rocket_node_idx = 0;


void classify_part(const time_series_type_t *in, float *out)
{
    /* Compute PPV features for this node's kernel subset */
    conv_multiple(in, features, get_kernels(), NUM_KERNELS, get_dilations(),
                  NUM_DILATIONS, get_biases(), NUM_BIASES_PER_KERNEL);

    /* Compute partial logits: weight * features + bias */
    for (uint32_t row_weight = 0; row_weight < NUM_CLASSES; row_weight++) {
        out[row_weight] = 0;
        for (uint32_t col_weight = 0; col_weight < devices_num_features[rocket_node_idx]; col_weight++) {
            out[row_weight] += features[col_weight] *
                               weight[row_weight * devices_num_features[rocket_node_idx] + col_weight];
        }
        out[row_weight] += bias[row_weight];
    }
}

static void calculate_exp(float *in, float *out, uint16_t length)
{
    for (uint16_t i = 0; i < length; i++) {
        out[i] = expf(in[i]);
    }
}

static float calculate_sum(float *in, uint16_t length)
{
    float result = 0;
    for (uint16_t i = 0; i < length; i++) {
        result += in[i];
    }
    return result;
}

uint8_t get_max_idx(float *in, uint8_t length)
{
    uint8_t best_idx = 0;
    float best_val = in[0];
    for (uint8_t i = 1; i < length; i++) {
        if (best_val < in[i]) {
            best_val = in[i];
            best_idx = i;
        }
    }
    return best_idx;
}

static void get_softmax(float *in, float *out)
{
    calculate_exp(in, out, NUM_CLASSES);
    float sum_exp = calculate_sum(out, NUM_CLASSES);
    for (uint8_t i = 0; i < NUM_CLASSES; i++) {
        out[i] /= sum_exp;
    }
}

uint8_t calculate_and_accumulate_gradient(float *out_pred, uint8_t idx_class)
{
    get_softmax(out_pred, out_softmax);

    float temp = 0;
    for (uint32_t row_weight = 0; row_weight < NUM_CLASSES; row_weight++) {
        for (uint32_t col_weight = 0; col_weight < devices_num_features[rocket_node_idx]; col_weight++) {
            temp = out_softmax[row_weight];
            if (row_weight == idx_class) {
                temp -= 1;
            }
            temp *= features[col_weight];
            d_weight[row_weight * devices_num_features[rocket_node_idx] + col_weight] += temp;
        }

        temp = out_softmax[row_weight];
        if (row_weight == idx_class) {
            temp -= 1;
        }
        d_bias[row_weight] += temp;
    }
    num_d++;
    return (get_max_idx(out_softmax, NUM_CLASSES) == idx_class);
}

static inline float float_abs(float f)
{
    return f > 0 ? f : -f;
}

static inline float float_max(float f1, float f2)
{
    return f1 > f2 ? f1 : f2;
}

#if USE_QADAM
static float calculate_qadam_scalings(float *array, uint16_t length)
{
    float scaling = 0;
    for (uint16_t i = 0; i < length; i++) {
        scaling = float_max(scaling, float_abs(array[i]));
    }
    return 1.0 / scaling;
}

static float rescale_array(float *array, uint32_t length, float scaling)
{
    for (uint32_t i = 0; i < length; i++) {
        array[i] *= scaling;
    }
    return 0;
}

static void qadam_step(float *params, float *d, uint8_t *m, uint8_t *v,
                       float *m_scalings, float *v_scalings,
                       uint32_t num_params, uint32_t num_scalings)
{
    float lr_t = LEARNING_RATE * sqrtf(1.0f - powf(BETA_2, t)) / (1.0f - powf(BETA_1, t));
    uint32_t param_idx = 0;

    rescale_array(d, num_params, 1.0 / num_d);

    for (uint16_t scaling_idx = 0; scaling_idx < num_scalings; scaling_idx++) {
        int remaining_params = num_params - param_idx;
        remaining_params = remaining_params < QADAM_BUFFER_SIZE ? remaining_params : QADAM_BUFFER_SIZE;

        dynamic_tree_dequantization(m + param_idx, block_wise_quantization_m_buffer, remaining_params);
        dynamic_tree_dequantization(v + param_idx, block_wise_quantization_v_buffer, remaining_params);

        float s = 1 / (m_scalings[scaling_idx] + 1e-7);
        rescale_array(block_wise_quantization_m_buffer, remaining_params, s);
        rescale_array(block_wise_quantization_v_buffer, remaining_params, 1 / (v_scalings[scaling_idx] + 1e-7));

        for (uint32_t i = 0; i < (uint32_t)remaining_params; i++) {
            block_wise_quantization_m_buffer[i] = BETA_1 * block_wise_quantization_m_buffer[i] + (1 - BETA_1) * d[param_idx + i];
            block_wise_quantization_v_buffer[i] = BETA_2 * block_wise_quantization_v_buffer[i] + (1 - BETA_2) * d[param_idx + i] * d[param_idx + i];
            params[param_idx + i] -= lr_t * block_wise_quantization_m_buffer[i] / (sqrtf(block_wise_quantization_v_buffer[i]) + EPSILON);
            d[param_idx + i] = 0;
        }

        m_scalings[scaling_idx] = calculate_qadam_scalings(block_wise_quantization_m_buffer, remaining_params);
        v_scalings[scaling_idx] = calculate_qadam_scalings(block_wise_quantization_v_buffer, remaining_params);

        rescale_array(block_wise_quantization_m_buffer, remaining_params, m_scalings[scaling_idx]);
        rescale_array(block_wise_quantization_v_buffer, remaining_params, v_scalings[scaling_idx]);

        dynamic_tree_quantization(block_wise_quantization_m_buffer, m + param_idx, remaining_params);
        dynamic_tree_quantization(block_wise_quantization_v_buffer, v + param_idx, remaining_params);

        param_idx += remaining_params;
        if (param_idx >= num_params) {
            break;
        }
    }
}
#endif


void update_weights(void)
{
#if USE_QADAM
    qadam_step(weight, d_weight, m_t_weight, v_t_weight,
               m_t_weight_scalings, v_t_weight_scalings,
               NUM_CLASSES * devices_num_features[rocket_node_idx], NUM_WEIGHT_SCALINGS);
    qadam_step(bias, d_bias, m_t_bias, v_t_bias,
               m_t_bias_scalings, v_t_bias_scalings,
               NUM_CLASSES, NUM_BIAS_SCALINGS);
#else
    float lr_t = LEARNING_RATE * sqrtf(1.0f - powf(BETA_2, t)) / (1.0f - powf(BETA_1, t));
    for (uint32_t row_weight = 0; row_weight < NUM_CLASSES; row_weight++) {
        for (uint32_t col_weight = 0; col_weight < devices_num_features[rocket_node_idx]; col_weight++) {
            uint32_t i = row_weight * devices_num_features[rocket_node_idx] + col_weight;
            d_weight[i] /= num_d;
            d_weight[i] += WEIGHT_DECAY * weight[i];
            m_t_weight[i] = BETA_1 * m_t_weight[i] + (1 - BETA_1) * d_weight[i];
            v_t_weight[i] = BETA_2 * v_t_weight[i] + (1 - BETA_2) * d_weight[i] * d_weight[i];
            weight[i] -= lr_t * m_t_weight[i] / (sqrtf(v_t_weight[i]) + EPSILON);
            d_weight[i] = 0;
        }
        d_bias[row_weight] /= num_d;
        d_bias[row_weight] += WEIGHT_DECAY * d_bias[row_weight];
        m_t_bias[row_weight] = BETA_1 * m_t_bias[row_weight] + (1 - BETA_1) * d_bias[row_weight];
        v_t_bias[row_weight] = BETA_2 * v_t_bias[row_weight] + (1 - BETA_2) * d_bias[row_weight] * d_bias[row_weight];
        bias[row_weight] -= lr_t * m_t_bias[row_weight] / (sqrtf(v_t_bias[row_weight]) + EPSILON);
        d_bias[row_weight] = 0;

        /* Only node 0 handles bias (others zero it) */
        if (rocket_node_idx != 0) {
            bias[row_weight] = 0;
        }
    }
#endif
    t++;
    num_d = 0;
}


void init_linear_classifier(uint8_t id)
{
    t = 1;
    /* Map the TOS_NODE_ID to 0-based rocket node index */
    rocket_node_idx = id - 1;

    for (uint32_t i = 0; i < NUM_CLASSES * devices_num_features[rocket_node_idx]; i++) {
        weight[i] = 2 * random_numbers[i % 1000] / sqrtf(NUM_FEATURES + NUM_CLASSES);
        d_weight[i] = 0;
    }

    for (uint32_t i = 0; i < NUM_CLASSES; i++) {
        bias[i] = 0;
        d_bias[i] = 0;
    }

#if USE_QADAM
    init_dynamic_tree_quantization();
    for (uint32_t i = 0; i < NUM_BIAS_SCALINGS; i++) {
        m_t_bias_scalings[i] = 1.0f;
        v_t_bias_scalings[i] = 1.0f;
    }
    for (uint32_t i = 0; i < NUM_WEIGHT_SCALINGS; i++) {
        m_t_weight_scalings[i] = 1.0f;
        v_t_weight_scalings[i] = 1.0f;
    }
    for (uint32_t i = 0; i < NUM_CLASSES * MAX_FEATURES_PER_DEVICE; i++) {
        m_t_weight[i] = 120;
        v_t_weight[i] = 120;
    }
    for (uint32_t i = 0; i < NUM_CLASSES; i++) {
        m_t_bias[i] = 120;
        v_t_bias[i] = 120;
    }
#else
    for (uint32_t i = 0; i < NUM_CLASSES * MAX_FEATURES_PER_DEVICE; i++) {
        m_t_weight[i] = 0;
        v_t_weight[i] = 0;
    }
    for (uint32_t i = 0; i < NUM_CLASSES; i++) {
        m_t_bias[i] = 0;
        v_t_bias[i] = 0;
    }
#endif
}
