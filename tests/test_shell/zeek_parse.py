from time import sleep
import os
import subprocess

def walk(rootPath):
    fileList = []
    for root, dirs, files in os.walk(rootPath):
        for name in files:
            fileList.append(os.path.join(root, name))
    return fileList

testPath = '/mnt/hgfs/Documents/pcap/'
pcapList = walk(testPath)
pcapNum = len(pcapList)

i = 1
for pcapFile in pcapList:
    if pcapFile[-4:] == 'pcap':
        homePath = pcapFile.rsplit('/',1)[0]
        # cc = 'zeek -r '+pcapFile+' /usr/local/zeek/var/lib/zkg/clones/package/spl-spt/scripts'
        # cc = 'zeek -r '+pcapFile+' /usr/local/zeek/share/zeek/site/flowmeter'
        cc = 'zeek local -r '+pcapFile
        try:
            subprocess.check_output(cc, shell=True, cwd=homePath,)
        except:
            print(pcapFile+'eeeeeeeeeeeeeerror')

        logFiles = walk(homePath)        
        for logFile in logFiles:
            if logFile[-3:] == 'log' :
                cc = 'mv '+ logFile + ' ' + pcapFile + '_' + logFile.rsplit('/',1)[1] + 'a' #logg avoid mv conflict
                subprocess.check_output(cc, shell=True) 
    i += 1
    # print(pcapFile)
    print(i/pcapNum)

 