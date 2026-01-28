#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /user/masters/OkeeffeCJ/PycharmProjects/Sem2LinuxFYP/my_pynq_project/myproject_prj/solution1/.autopilot/db/a.g.bc ${1+"$@"}
