# Copyright 2021 Nokia
#
# Licensed under the Apache License 2.0
# SPDX-License-Identifier: Apache-2.0

*** Settings ***
Documentation     Test suite for making the configuration of a test cluster; it should include all resources that any configuration setting would need; this is an empty suite

Test Timeout      5 minutes

Force Tags        this_is_not_a_real_tc


*** Test Cases ***
FakeTC
    [Documentation]    It is here so that this suite is parsed properly
    [Tags]    remove
    Fail    I should never run

*** Keywords ***
Set param1
    [Arguments]    ${value}
    [Documentation]    This keyword sets configuration parameter param1 to all defined values
    Log    param1 set to ${value}
    # Do something meaningful here

Set param2
    [Arguments]    ${value}
    [Documentation]    This keyword sets configuration parameter param2 to all defined values
    Log    param2 set to ${value}
    # Do something meaningful here

Set param3
    [Arguments]    ${value}
    [Documentation]    This keyword sets configuration parameter param3 to all defined values
    Log    param2 set to ${value}
    # Do something meaningful here

Set param4
    [Arguments]    ${value}
    [Documentation]    This keyword sets configuration parameter param4 to all defined values
    Log    param2 set to ${value}
    # Do something meaningful here
