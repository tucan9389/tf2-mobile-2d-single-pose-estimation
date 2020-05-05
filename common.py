# Copyright 2019 Doyoung Gwak (tucan.dev@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ======================
#-*- coding: utf-8 -*-

import datetime

def get_time_and_step_interval(current_step, is_init=None):
    global before_step_and_time
    _current_step = current_step

    if is_init:
        before_step_and_time = (_current_step, datetime.datetime.now())
        return

    _before_step, _before_time = before_step_and_time
    _current_time = datetime.datetime.now()
    before_step_and_time = (_current_step, _current_time)

    _step_interval = _current_step - _before_step
    _elapsed_time_interval = _current_time - _before_time

    _time_interval_str = get_time_to_str(_elapsed_time_interval.total_seconds())
    _time_interval_per_step_str = get_time_to_str(_elapsed_time_interval.total_seconds() / float(_step_interval))

    return _time_interval_str, _time_interval_per_step_str

def get_time_to_str(total_seconds):
    if int(total_seconds) == 0:
        if int(total_seconds * 1000.) == 0:
            return "%.3fms" % (total_seconds * 1000.)
        else:
            return "%.3fs" % (total_seconds)
    elif total_seconds <= 60:
        return "%ds" % int(total_seconds)
    else:
        m, s = divmod(total_seconds, 60)
        return "%dm %02ds" % (m, int(s))