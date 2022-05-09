#!/bin/sh

#subjectName=$1
imgDir=$pwd
#/mnt/Data01/'`date +%Y%m%d`'.'$subjectName'.'$subjectName'/'
delta=0.0001  #seconds

stamp=`gdate +%Y%m%d%H%M%S`
echo $imgDir

scanNum=3
longScanNum=$(seq -f "%02g" $scanNum $scanNum)


for fileNum in $(seq -f "%03g" 1 999)
do
    fileName=$imgDir'001_0000'$longScanNum'_000'$fileNum.dcm
    echo awaiting $fileName
    while [ ! -f $fileName ]
    do
	#echo 1
        sleep $delta
    done
    echo .
    #echo `gdate +%s.%N  | cut -b1-13` $fileName
    echo `gdate +%s.%N ` >> ~/bridge/code/rtutils/timing_logs/file_arrival_timing_poll_$stamp.txt
done

