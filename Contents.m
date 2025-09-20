% Contents.m
% 
%   This folder contains MATLAB code to accompany the paper:
%
%   "Automatic nonstationary anisotropic Tikhonov regularization 
%    through bilevel optimization" - Gazzola and Gholami, arXive, 2024.
%
% These codes require the L-BFGS-B-C functions: https://github.com/stephenbeckr/L-BFGS-B-C
%
%% Example script.
%
% test_denoising.m - reproduces the denoising test problem in the paper,
%                    with the Clapp model (saved in Clapp_model_resized.mat)
%                    as ground-truth.
%
%% Main files.
% smoothulwithgrad    - computes the upper level objective function for the
%                       fully anisotropic model, and its gradient
% smoothulwithgrad_pa - computes the upper level objective function for the
%                       partially anisotropic model, and its gradient
% smoothDPwithgrad    - computes the upper level objective function for saftisfying 
%                       the discrepancy principle, and its gradient