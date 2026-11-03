set script_dir [file dirname [file normalize [info script]]]
cd [file normalize [file join $script_dir ..]]
set project_dir [file normalize [file join $script_dir .. hls_project_r9x4]]
open_project -reset $project_dir
file copy -force [file join $script_dir ecg_nn_accel.cpp] [file join $project_dir ecg_nn_accel.cpp]
file copy -force [file join $script_dir ecg_nn_params.h] [file join $project_dir ecg_nn_params.h]
set_top ecg_nn_accel
add_files [file join $project_dir ecg_nn_accel.cpp] -cflags "-std=c++14"
add_files [file join $project_dir ecg_nn_params.h]
add_files -tb [file join $script_dir ecg_nn_tb.cpp] -cflags "-std=c++14"
open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design -argv [file join $script_dir .. golden_vectors]
csynth_design
exit
