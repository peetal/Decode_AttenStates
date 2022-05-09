#!/bin/bash

stamp=`date +%Y%m%d%H%M%S`

while true; do

   read -s -n 1 key  # -s: do not echo input character. -n 1: read only 1 character (separate with space)

   if [[ $key = "'" ]]; then 
      #echo `gdate +%s.%N  | cut -b1-13` >> pulse_timing_$stamp.txt
      echo `date +%s.%N ` >> pulse_timing_$stamp.txt
   fi

done
