# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "CS_HOLD_CYCLES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "POST_RESET_CYCLES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "POWER_WAIT_CYCLES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "PRE_RESET_CYCLES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "REF_WAIT_CYCLES" -parent ${Page_0}
  ipgui::add_param $IPINST -name "RESET_PULSE_CYCLES" -parent ${Page_0}


}

proc update_PARAM_VALUE.CS_HOLD_CYCLES { PARAM_VALUE.CS_HOLD_CYCLES } {
	# Procedure called to update CS_HOLD_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CS_HOLD_CYCLES { PARAM_VALUE.CS_HOLD_CYCLES } {
	# Procedure called to validate CS_HOLD_CYCLES
	return true
}

proc update_PARAM_VALUE.POST_RESET_CYCLES { PARAM_VALUE.POST_RESET_CYCLES } {
	# Procedure called to update POST_RESET_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.POST_RESET_CYCLES { PARAM_VALUE.POST_RESET_CYCLES } {
	# Procedure called to validate POST_RESET_CYCLES
	return true
}

proc update_PARAM_VALUE.POWER_WAIT_CYCLES { PARAM_VALUE.POWER_WAIT_CYCLES } {
	# Procedure called to update POWER_WAIT_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.POWER_WAIT_CYCLES { PARAM_VALUE.POWER_WAIT_CYCLES } {
	# Procedure called to validate POWER_WAIT_CYCLES
	return true
}

proc update_PARAM_VALUE.PRE_RESET_CYCLES { PARAM_VALUE.PRE_RESET_CYCLES } {
	# Procedure called to update PRE_RESET_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.PRE_RESET_CYCLES { PARAM_VALUE.PRE_RESET_CYCLES } {
	# Procedure called to validate PRE_RESET_CYCLES
	return true
}

proc update_PARAM_VALUE.REF_WAIT_CYCLES { PARAM_VALUE.REF_WAIT_CYCLES } {
	# Procedure called to update REF_WAIT_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.REF_WAIT_CYCLES { PARAM_VALUE.REF_WAIT_CYCLES } {
	# Procedure called to validate REF_WAIT_CYCLES
	return true
}

proc update_PARAM_VALUE.RESET_PULSE_CYCLES { PARAM_VALUE.RESET_PULSE_CYCLES } {
	# Procedure called to update RESET_PULSE_CYCLES when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.RESET_PULSE_CYCLES { PARAM_VALUE.RESET_PULSE_CYCLES } {
	# Procedure called to validate RESET_PULSE_CYCLES
	return true
}


proc update_MODELPARAM_VALUE.POWER_WAIT_CYCLES { MODELPARAM_VALUE.POWER_WAIT_CYCLES PARAM_VALUE.POWER_WAIT_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.POWER_WAIT_CYCLES}] ${MODELPARAM_VALUE.POWER_WAIT_CYCLES}
}

proc update_MODELPARAM_VALUE.PRE_RESET_CYCLES { MODELPARAM_VALUE.PRE_RESET_CYCLES PARAM_VALUE.PRE_RESET_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.PRE_RESET_CYCLES}] ${MODELPARAM_VALUE.PRE_RESET_CYCLES}
}

proc update_MODELPARAM_VALUE.RESET_PULSE_CYCLES { MODELPARAM_VALUE.RESET_PULSE_CYCLES PARAM_VALUE.RESET_PULSE_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.RESET_PULSE_CYCLES}] ${MODELPARAM_VALUE.RESET_PULSE_CYCLES}
}

proc update_MODELPARAM_VALUE.POST_RESET_CYCLES { MODELPARAM_VALUE.POST_RESET_CYCLES PARAM_VALUE.POST_RESET_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.POST_RESET_CYCLES}] ${MODELPARAM_VALUE.POST_RESET_CYCLES}
}

proc update_MODELPARAM_VALUE.REF_WAIT_CYCLES { MODELPARAM_VALUE.REF_WAIT_CYCLES PARAM_VALUE.REF_WAIT_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.REF_WAIT_CYCLES}] ${MODELPARAM_VALUE.REF_WAIT_CYCLES}
}

proc update_MODELPARAM_VALUE.CS_HOLD_CYCLES { MODELPARAM_VALUE.CS_HOLD_CYCLES PARAM_VALUE.CS_HOLD_CYCLES } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CS_HOLD_CYCLES}] ${MODELPARAM_VALUE.CS_HOLD_CYCLES}
}

