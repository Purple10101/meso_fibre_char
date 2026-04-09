"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Joshua Poole
20260409

config.py
system configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# switches
SS5_ENABLED = True
SS6_ENABLED = False

# dependency list
READY_DEPENDENCIES = {"ss4"}
if SS5_ENABLED:
    READY_DEPENDENCIES.add("ss5")
if SS6_ENABLED:
    READY_DEPENDENCIES.add("ss6")

# specializations list
DOWNSTREAM_PROCESSORS = set()
if SS5_ENABLED:
    DOWNSTREAM_PROCESSORS.add("ss5")
if SS6_ENABLED:
    READY_DEPENDENCIES.add("ss6")