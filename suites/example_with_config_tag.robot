# Copyright 2021 Nokia
#
# Licensed under the Apache License 2.0
# SPDX-License-Identifier: Apache-2.0

*** Test Cases ***
001 Test case with single config entry
    [Tags]    config-param1=whatever
    Log to console    I am testcase ${TEST_NAME}
    Sleep    2s

002 Test case without any config entry
    Log to console    I am testcase ${TEST_NAME}
    Sleep    1s

003 Test case with two config entries
    [Tags]    config-param1=whatever
    ...       config-param2=internal
    Log to console    I am testcase ${TEST_NAME}
    Sleep    1s

004 Test case with different value for config-param2
    [Tags]    config-param2=external
    Log to console    I am testcase ${TEST_NAME}
    Sleep    5s

005 Test case with two config entries
    [Tags]    config-param2=external
    ...      config-param3=False
    Log to console    I am testcase ${TEST_NAME}
    Sleep    5s

006 Test case with three config entries
    [Tags]    config-param2=internal
    ...       config-param3=True
    ...       config-param4=what
    Log to console    I am testcase ${TEST_NAME}
    Sleep    2s

*** Keywords ***
Keyword
    [Arguments]    ${arg}
    Log    ${arg}
