#pragma once

//------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------
namespace Parameters
{
    const double timestep = 1.0;

    const unsigned int input_size = 32;
    const unsigned int centre_size = 20;

    const double convergent_scale =  ((16.0 * 16.0) / ((double)centre_size * (double)centre_size));

    const double persistance_e = 0.1;
    const double persistance_i = 0.8;
    const double persistance_s = 0.4;
    const double persistance_f = 0.1;
    const double persistance_lgmd = 0.4;

    const double i_s_weight_scale = 0.2 * 2.0 * 0.04;

    const unsigned int i_s_delay_1 = 3;
    const double i_s_weight_1 = -0.4 * i_s_weight_scale;

    const unsigned int i_s_delay_2 = 3;
    const double i_s_weight_2 = -0.32 * i_s_weight_scale;

    const unsigned int i_s_delay_4 = 4;
    const double i_s_weight_4 = -0.2 * i_s_weight_scale;
}