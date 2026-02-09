Final Year Project for Masters in Electrical and Electronic Engineering at University College Cork

Project scope: To implement a small neural network, previously developed by another UCC summer intern for use on a Qualcomm RB3, and use hls4ml to implement this network on a PYNQ Z2 FPGA board.

Update: Deployment on PYNQ Z2 ran into issues with hls4ml and Vivado HLS compilers, due to large reuse factor. Design did not synthesize. Moving to Alveo U280.