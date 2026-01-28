#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_1[N_INPUT_1_1],
    result_t layer10_out[N_LAYER_10]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer10_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<dense_weight_t, 600>(w2, "w2.txt");
        nnet::load_weights_from_txt<dense_bias_t, 100>(b2, "b2.txt");
        nnet::load_weights_from_txt<dense_1_weight_t, 10000>(w4, "w4.txt");
        nnet::load_weights_from_txt<dense_1_bias_t, 100>(b4, "b4.txt");
        nnet::load_weights_from_txt<dense_2_weight_t, 10000>(w6, "w6.txt");
        nnet::load_weights_from_txt<dense_2_bias_t, 100>(b6, "b6.txt");
        nnet::load_weights_from_txt<dense_3_weight_t, 10000>(w8, "w8.txt");
        nnet::load_weights_from_txt<dense_3_bias_t, 100>(b8, "b8.txt");
        nnet::load_weights_from_txt<output_1_weight_t, 100>(w10, "w10.txt");
        nnet::load_weights_from_txt<output_1_bias_t, 1>(b10, "b10.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    dense_result_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, dense_result_t, config2>(input_1, layer2_out, w2, b2); // dense

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<dense_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // dense_relu

    dense_1_result_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, dense_1_result_t, config4>(layer3_out, layer4_out, w4, b4); // dense_1

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<dense_1_result_t, layer5_t, relu_config5>(layer4_out, layer5_out); // dense_1_relu

    dense_2_result_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, dense_2_result_t, config6>(layer5_out, layer6_out, w6, b6); // dense_2

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<dense_2_result_t, layer7_t, relu_config7>(layer6_out, layer7_out); // dense_2_relu

    dense_3_result_t layer8_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::dense<layer7_t, dense_3_result_t, config8>(layer7_out, layer8_out, w8, b8); // dense_3

    layer9_t layer9_out[N_LAYER_8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<dense_3_result_t, layer9_t, relu_config9>(layer8_out, layer9_out); // dense_3_relu

    nnet::dense<layer9_t, result_t, config10>(layer9_out, layer10_out, w10, b10); // output_1

}

