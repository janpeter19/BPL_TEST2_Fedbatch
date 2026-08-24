# setup applicateion data BPL_TEST2_Fedbatch 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-21 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import matplotlib.pyplot as plt 
from pyfmi import load_fmu

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
global model
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   flag_vendor = 'JM'
   flag_type = 'CS'
   fmu_model ='BPL_TEST2_Fedbatch_windows_jm_cs.fmu'        
   model = load_fmu(fmu_model, log_level=0)  
elif platform.system() == 'Linux': 
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_TEST2_Fedbatch_linux_om_cs.fmu'    
         model = load_fmu(fmu_model, log_level=0) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_TEST2_Fedbatch_linux_om_me.fmu'   
#        fmu_model ='BPL_TEST2_Fedbatch_linux_2404_om_me.fmu'   
         model = load_fmu(fmu_model, log_level=0)
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = model.simulate_options()
   opts_std['silent_mode'] = True
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'     
elif flag_type in ['ME', 'me']:
   opts_std = model.simulate_options()
   opts_std["CVode_options"]["verbosity"] = 50 
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'  
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   MSL_usage = model.get('MSL.usage')[0]
   MSL_version = model.get('MSL.version')[0]
   BPL_version = model.get('BPL.version')[0]
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '4.1.0 - used components: RealInput, RealOutput' 
   MSL_version = '4.1.0'
   BPL_version = 'Bioprocess Library version 2.3.2' 
else:    
   print('There is no FMU for this platform')

# Simulation time
simulationTime = 5.0
prevFinalTime = 0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Create stateValue that later will be used to store final state and used for initialization in 'cont':
stateValue =  {}
stateValue = model.get_states_list()
stateValue.update(timeDiscreteStates)

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture']

# Provide process diagram on disk
fmu_process_diagram ='BPL_TEST2_Fedbatch_process_diagram_om.png'

#------------------------------------------------------------------------------------------------------------------

# Create dictionaries parValue[] and parLocation[]
parValue = {}
parValue['V_start'] = 1.0
parValue['VX_start'] = 1.0
parValue['VS_start'] = 10.0

parValue['Y'] = 0.5
parValue['qSmax'] = 1.0
parValue['Ks'] = 0.1

parValue['feedtank.S_in'] = 300.0
parValue['feedtank.V_start'] = 10.0
parValue['F_start'] = 0
parValue['mu_feed'] = 0.10
parValue['t_startExp'] = 3.0
parValue['F_startExp'] = 1.33e-3
parValue['F_max'] = 0.3

parLocation = {}
parLocation['V_start'] = 'bioreactor.V_start'
parLocation['VX_start'] = 'bioreactor.m_start[1]' 
parLocation['VS_start'] = 'bioreactor.m_start[2]' 

parLocation['Y'] = 'bioreactor.culture.Y'
parLocation['qSmax'] = 'bioreactor.culture.qSmax'
parLocation['Ks'] = 'bioreactor.culture.Ks'

parLocation['feedtank.S_in'] = 'feedtank.c_in[2]'
parLocation['feedtank.V_start'] = 'feedtank.V_start'
parLocation['F_start'] = 'dosagescheme.F_start'
parLocation['mu_feed'] = 'dosagescheme.mu_feed'
parLocation['t_startExp'] = 'dosagescheme.t_startExp'
parLocation['F_startExp'] = 'dosagescheme.F_startExp'
parLocation['F_max'] = 'dosagescheme.F_max'

# Extra for describe()
parLocation['mu'] = 'bioreactor.culture.mu'

# Parameter value check - especially for hysteresis to avoid runtime error
parCheck = []
parCheck.append("parValue['Y'] > 0")
parCheck.append("parValue['qSmax'] > 0")
parCheck.append("parValue['Ks'] > 0")
parCheck.append("parValue['V_start'] > 0")
parCheck.append("parValue['VX_start'] >= 0")
parCheck.append("parValue['VS_start'] >= 0")
parCheck.append("parValue['t_startExp'] >= 0")

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: parValue, parLocation, parCheck, diagrams, ax, lines
#------------------------------------------------------------------------------------------------------------------

# Create list of diagrams to be plotted by simu()
diagrams = []

# Create an empty list axes to be defined in newplot() and plotted by simu() or show()
ax = []

# Create list of pens for the diagrams
lines = ['-','--',':','-.']