set root [file normalize [file join [file dirname [info script]] ../..]]
set src [file join $root PYNQ-3.1.1 boards Pynq-Z2 ecg_nn_r9q]
set dst [file join $root PYNQ-3.1.1 boards Pynq-Z2 ecg_nn_r9qf]
open_project [file join $src ecg_nn_r9q.xpr]
save_project_as -exclude_run_results ecg_nn_r9qf $dst
close_project
exit
