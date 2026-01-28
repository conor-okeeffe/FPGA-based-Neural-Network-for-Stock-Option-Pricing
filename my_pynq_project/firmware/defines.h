#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 6
#define N_LAYER_2 100
#define N_LAYER_2 100
#define N_LAYER_4 100
#define N_LAYER_4 100
#define N_LAYER_6 100
#define N_LAYER_6 100
#define N_LAYER_8 100
#define N_LAYER_8 100
#define N_LAYER_10 1


// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<36,16> dense_result_t;
typedef ap_fixed<16,6> dense_weight_t;
typedef ap_fixed<16,6> dense_bias_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> dense_relu_table_t;
typedef ap_fixed<40,20> dense_1_result_t;
typedef ap_fixed<16,6> dense_1_weight_t;
typedef ap_fixed<16,6> dense_1_bias_t;
typedef ap_uint<1> layer4_index;
typedef ap_fixed<16,6> layer5_t;
typedef ap_fixed<18,8> dense_1_relu_table_t;
typedef ap_fixed<40,20> dense_2_result_t;
typedef ap_fixed<16,6> dense_2_weight_t;
typedef ap_fixed<16,6> dense_2_bias_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> dense_2_relu_table_t;
typedef ap_fixed<40,20> dense_3_result_t;
typedef ap_fixed<16,6> dense_3_weight_t;
typedef ap_fixed<16,6> dense_3_bias_t;
typedef ap_uint<1> layer8_index;
typedef ap_fixed<16,6> layer9_t;
typedef ap_fixed<18,8> dense_3_relu_table_t;
typedef ap_fixed<40,20> result_t;
typedef ap_fixed<16,6> output_1_weight_t;
typedef ap_fixed<16,6> output_1_bias_t;
typedef ap_uint<1> layer10_index;


#endif
