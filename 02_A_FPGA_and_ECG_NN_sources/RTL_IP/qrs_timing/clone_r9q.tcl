set root [file normalize [file join [file dirname [info script]] ../..]]
set src [file join $root PYNQ-3.1.1 boards Pynq-Z2 ecg_nn_r9]
set dst [file join $root PYNQ-3.1.1 boards Pynq-Z2 ecg_nn_r9q]
open_project [file join $src ecg_nn_r9.xpr]
save_project_as -exclude_run_results ecg_nn_r9q $dst
close_project
exit
