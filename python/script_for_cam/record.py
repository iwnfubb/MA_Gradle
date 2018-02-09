import os,sys
import requests
import traceback
import time
from configparser import ConfigParser
from threading import Thread
import telnetlib
import numpy as np
import queue

import logging as log

log.basicConfig(stream=sys.stdout,level=log.INFO, format='%(asctime)s %(message)s')


import win32api,win32process,win32con
pid = win32api.GetCurrentProcessId()
handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)

########## DEFINES #############
#mode ="runAsStream"
#mode ="record"
mode ="recordForEver"
global terminateFlag
terminateFlag = False
configFile = "settings.ini"
ALARM_0_IDENTIFIER = "AlarmTextSetState io_id 0x03230000 onoff"
ALARM_1_IDENTIFIER = "AlarmTextSetState io_id 0x0323ffff onoff"
ALARM_LIST = [ALARM_0_IDENTIFIER, ALARM_1_IDENTIFIER]
STREAM_BREAK_TIMEOUT = 180 # 3min
DET_STREAM_BREAK_TIMEOUT = 20 # 20sec

PIR_URL = "http://%s/rcp.xml?command=0x0c1b&type=F_FLAG&num=%d&direction=READ"
PRINT_TERMINATOR = "!EXIT!"

global alarm_flag_0, alarm_flag_1, streaming_started_flag
alarm_flag_0 = False
alarm_flag_1 = False
streaming_started_flag = False
STREAM_CONNECTION_TIMEOUT = 10.

global queue_PIR_in, print_queue
queue_PIR_in = queue.Queue()
print_queue = queue.Queue()

def setup_logger(name, log_file, level=log.INFO):
    """Function setup as many loggers as you want"""

    handler = log.FileHandler(log_file)
    formatter = log.Formatter('%(message)s')        
    handler.setFormatter(formatter)

    logger = log.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger, handler

class PrintMetaThread(Thread):
    def __init__(self, queue, filenameOutput):
        Thread.__init__(self)
        self.queue = queue
        self.logger, self.handler = setup_logger('meta_data', filenameOutput)

    def run(self):
        global terminateFlag
        while not terminateFlag:
            result = self.queue.get()
            if result.count(PRINT_TERMINATOR)>0:
                break
            #log.info("queue_el: %s" %str(result))
            self.logger.info(result)
            self.queue.task_done()
        self.logger.removeHandler(self.handler)

class ReadPirThread(Thread):
    def __init__(self, queue_PIR_in, print_queue, ip, user="service", password = ""):
        Thread.__init__(self)
        self.queue_PIR_in = queue_PIR_in
        self.print_queue = print_queue
        self.ip = ip
        self.user = user
        self.password = password

    def getPIRState(self, pirID, timeStamp):
        assert pirID in [1,2,3,], "Wrong PIR id"
        url = PIR_URL%(self.ip,pirID)
        response = requests.get(url, auth=(self.user, self.password), timeout=10.)
        content = response.content.decode(encoding='utf_8', errors='strict')
        output = "PIR%d=%s" %(pirID, content.split('<result>')[-1].split('<dec>')[-1].split("</dec>")[0].strip())
        response.close()

        newLine = "%s - %s" %(timeStamp, output)
        self.print_queue.put(newLine)

    def run(self):
        global terminateFlag
        while not terminateFlag:
            pirId, timeStamp = self.queue_PIR_in.get()
            if pirId <0:
                break;
            self.getPIRState(pirId, timeStamp)
            self.queue_PIR_in.task_done()

def streamFromCam(ip, local_filename, qualityLevel = 1, user="service", password = ""):
    global terminateFlag, streaming_started_flag
    cam2 = "http://%s/video.mp4?inst=%d&enableaudio=0"%(ip,qualityLevel)
    log.info("Opening stream: %s" , cam2)
    stream = None
    try:
        stream = requests.get(cam2, auth=(user, password), stream = True, timeout=STREAM_CONNECTION_TIMEOUT)
    except requests.exceptions.Timeout:
        # Maybe set up for a retry, or continue in a retry loop
        log.error("Timeout opening stream: %s" , cam2)
        return
    except requests.exceptions.TooManyRedirects:
        # Tell the user their URL was bad and try a different one
        log.error("TooManyRedirects opening stream: %s" , cam2)
        return
    except requests.exceptions.RequestException as e:
        # catastrophic error. bail.
        log.error("catastrophic error opening stream: %s, %s" , cam2, e)
        return


    if terminateFlag:
        if stream != None:
            stream.close()
        # is this needed?
        return

    streaming_started_flag = True
    try:
        with open(local_filename, 'wb') as f:
                while True:
                    buf = stream.raw.read(16*1024)
                    if not buf or terminateFlag:
                        break
                    f.write(buf)
        stream.close()
    except:
        log.warning("Video stream is broken down...")
    streaming_started_flag = False
    if stream != None:
        stream.close()

def setAlarms(line):
    global alarm_flag_0, alarm_flag_1
    if line.count(ALARM_0_IDENTIFIER)>0:
        value = bool(int(line.split(ALARM_0_IDENTIFIER)[-1].strip()))
        alarm_flag_0 = value
        if value:
            log.info('VCA Alarm = 1')
        else:
            log.info('VCA Alarm = 0')
        return
    elif line.count(ALARM_1_IDENTIFIER)>0:
        value = bool(int(line.split(ALARM_1_IDENTIFIER)[-1].strip()))
        alarm_flag_1 = value
        if value:
            log.debug('PIR Alarm = 1')
        else:
            log.debug('PIR Alarm = 0')


def contentAnalysis(content, timeStamp, printFlag = False):
    global queue_PIR_in
    if content.count("PIR Input " )>0:
        if printFlag:
            print("B:%s"%content)
        pirIndex = int(content.split("PIR Input " )[-1][0])
        queue_PIR_in.put((pirIndex, timeStamp))

def readFromCamWithTelnet(ipCam, port= 23, printFlag = False):
    global terminateFlag, streaming_started_flag, print_queue
    log.info("Waiting until streaming has start...")
    ticTime = time.time()
    while not streaming_started_flag:
        if time.time()-ticTime > (STREAM_CONNECTION_TIMEOUT + 5):
            return
        time.sleep(0.05)
    log.info("Starting telnet session, to read Alarms.")
    tn = telnetlib.Telnet(ipCam, port)
    firstContent = tn.read_until(b"\r\n").decode(encoding='utf_8')
    if (firstContent.count("only one telnet session supported") == 0) and not terminateFlag:
        try:
            content = ""
            while not terminateFlag:
                content += tn.read_until(b"\r\n", timeout=1).decode(encoding='utf_8')
                
                if len(content)>2 and content.endswith("\r\n"):
                    
                    timeStamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
                    newLine = "%s - %s" %(timeStamp, content[:-2])
                    setAlarms(newLine)
                    if printFlag:
                        print(newLine)
                    #log.info("Q1: " + newLine)
                    print_queue.put(newLine)
                    contentAnalysis(content[:-2], timeStamp, printFlag)
                    content = ""
        except Exception:
            log.error("Telnet connection to write metadata is broken down...")
            print(traceback.format_exc())
            # or
            print(sys.exc_info()[0])

        log.warning("Termination flag detected...")
    else:
        log.warning("Sorry session is blocked")
    tn.close()

def terminateThreads(maxNumRetries = 30):
    global terminateFlag, queue_PIR_in, print_queue
    log.info("Terminating...")
    # send signals to stop threads
    terminateFlag = True
    # Wait until main threads end
    counter = 0
    while (streamingThread.is_alive() or metaThread.is_alive()):
        if counter > maxNumRetries:
            break  
        time.sleep(1)
        counter += 1
    log.info("Main threads are shut down...")
    # finish working left over tasks for printing
    print_queue.put(PRINT_TERMINATOR)
    queue_PIR_in.put((-1,-1))
    while ( printThread.is_alive()  or pirThread.is_alive()):
        if counter > maxNumRetries:
            break
        time.sleep(1)
    
    assert not streamingThread.is_alive(), "Error terminating video streaming thread  (duration = %d)" %counter
    assert not metaThread.is_alive(), "Error terminating metadata reading thread  (duration = %d)" %counter
    assert not printThread.is_alive(), "Error terminating print thread  (duration = %d)" %counter
    assert not pirThread.is_alive(), "Error terminating pir thread  (duration = %d)" %counter
    queue_PIR_in.queue.clear()
    print_queue.queue.clear()
    print("Termination (duration = %d) successful! "%counter)


if __name__ == "__main__":

    cfg = ConfigParser()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    configFileFull = os.path.join(dir_path, configFile)
    if not os.path.isfile(configFileFull):
        log.info("Error opening config file %s" % configFileFull)
        sys.exit(1)
    cfg.read(configFileFull)
    qualityLevel = int(cfg.get("SETUP", "qualityLevel"))
    user = cfg.get("SETUP", "user")
    password = cfg.get("SETUP", "password")
    ip = cfg.get("SETUP", "ip")
    delay = float(cfg.get("SETUP", "delay")) if cfg.has_option("SETUP", "delay") else 0 
    
    recordingDirectory = cfg.get("SETUP", "recordingDirectory")
    mode = cfg.get("SETUP", "mode")


    while True:
        log.info('Waiting for %f seconds to start...', delay)
        time.sleep(delay)
        dayStamp = time.strftime("%Y%m%d", time.gmtime())
        timeStamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        recSubDir = os.path.join(recordingDirectory, dayStamp)
        if not os.path.isdir(recSubDir):
            os.makedirs(recSubDir)

        local_filename_base = "%s_rec_%d.mp4" %(timeStamp,qualityLevel)
        local_filename_meta_base = "%s_meta.dat" %(timeStamp)
        local_filename_pir_base = "%s_pir.dat" %(timeStamp)
        local_filename = os.path.join(recSubDir, local_filename_base)
        local_filename_meta = os.path.join(recSubDir, local_filename_meta_base)
        local_filename_pir = os.path.join(recSubDir, local_filename_pir_base)

        metaThread = Thread(target=readFromCamWithTelnet, args=(ip,))
        streamingThread = Thread(target=streamFromCam,args=(ip, local_filename, qualityLevel, user, password) )
        
        # spawn threads to print
        pirThread = ReadPirThread(queue_PIR_in, print_queue, ip, user, password )
        printThread = PrintMetaThread(print_queue, local_filename_meta)
        
        terminateFlag = False
        worthFlag = [False, False]
        
        # Start recording
        streamingThread.start()
        printThread.start()
        pirThread.start()
        metaThread.start()
        ticTime = time.time()
        
        while streamingThread.is_alive() and metaThread.is_alive() and pirThread.is_alive() and printThread.is_alive():
            if np.any([alarm_flag_0, alarm_flag_1]):
                ticTime = time.time()
                if alarm_flag_0:
                    worthFlag[0] = True
                if alarm_flag_1:
                    worthFlag[1] = True
                log.debug("A0,A1 = (%s, %s)" %(str(alarm_flag_0),str(alarm_flag_1)))
            if np.any(worthFlag) and ((time.time()-ticTime)> DET_STREAM_BREAK_TIMEOUT):
                log.warning("Stream Timeout (%d sec) was reached, exiting now!"%DET_STREAM_BREAK_TIMEOUT)
                break;
            elif (time.time()-ticTime)> STREAM_BREAK_TIMEOUT:
                log.warning("Stream Timeout (%d sec) was reached, exiting now!"%STREAM_BREAK_TIMEOUT)
                break;
            time.sleep(1)

        terminateThreads()
        log.info("Will restart in a moment...")
        #

        if not (mode == "recordForEver"):
            break