import serial
import struct
import argparse
import time 
import csv
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


PKT_LEN = 10
first_index = 0
last_index = 0
gap_cnt = 0
last_gap_cnt = 0
def parse_n_save_results(buf):
    global PKT_LEN
    global first_index
    global last_index
    global gap_cnt
    i=0
    while i + PKT_LEN < len(buf):
        if buf[i] != 0xAA and buf[i+1] != 0x55:
            i += 1
            continue
        pkt = buf[i:i+PKT_LEN]     
        _, _, cnt, err, ref = struct.unpack("<BBLHH", pkt)
        if first_index == 0:
            first_index = cnt
            
        cnt -= first_index
        # print(f"#{cnt} err: {err}, ref:{ref}")
        if last_index != cnt - 1:
            last_gap_cnt = gap_cnt
            gap_cnt += cnt - last_index
            print(f"Found {gap_cnt - last_gap_cnt} gaps")
        last_index = cnt
        with open('out_test/out.csv', 'a', newline='') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',', quotechar='|')
            datawriter.writerow([cnt, err, ref])
        
        i += PKT_LEN
        
        
    if i:
        del buf[:i]

def parse_stream(ser, duration):
    buf = bytearray()
    ch_adc = {}
    start = time.time()
    print(f"[+] capture start on {ser.port} @{ser.baudrate} baud")
    with open('out_test/out.csv', mode='w', newline='') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', quotechar='|')
        datawriter.writerow(["#", "error", "reference"])
    try:
        while True:
            chunk = ser.read(4096)
            buf += chunk
            parse_n_save_results(buf)
    except KeyboardInterrupt:
        print(f"KeyboardInterrupt")

    except Exception as inst:
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst) 
        return
    finally:
        ser.close()
        finish = time.time()
        print(f"Got {(finish - start)*1000} sec")
        print(f"Total {gap_cnt} gaps")
        ind = []
        err = []
        ref = []
        with open('out_test/out.csv', 'r', newline='') as csvfile:
            spamreader = csv.DictReader(csvfile)
            for row in spamreader:
                err.append(int(row['error']))
                ref.append(int(row['reference']))
                ind.append(int(row["#"]))
        print(f"Read {len(ind)} lines ({len(ind)*(1/8000)} sec)")
        time_axis = []
        for i in ind:
            time_axis.append((i*(1/16000))*1000)
            
        y_err = np.array(err, dtype=np.float32)
        y_ref = np.array(ref, dtype=np.float32)
        x_axis = np.array(time_axis)
        
        plt.subplot(2, 1, 1)
        plt.title("Error mic")
        plt.plot(x_axis, y_err)
        
        plt.subplot(2, 1, 2)
        plt.title("reference mic")
        plt.plot(x_axis, y_ref)
        plt.show()
        
        y_ref_snd = ((y_ref.astype(np.float32) / 4095) - 0.5) * 2  # нормализуем к [-1, 1]    
        # Нормализуем данные, если нужно (для float32 от -1.0 до 1.0)
        if y_ref_snd.dtype == np.float32:
            y_ref_snd = y_ref_snd / np.max(np.abs(y_ref_snd))
          
        print(len(y_ref_snd))
        sd.play(y_ref_snd, 8000)
        sd.wait()
        
        
        
        
        

def open_serial(port="COM1", baud=115200):
    ser = serial.Serial(port, baud, timeout=0,
                    parity=serial.PARITY_EVEN, rtscts=1)
    if ser.is_open is not True:
        raise Exception(f"Can't open port {port}", )
    return ser

def main():
    p = argparse.ArgumentParser(prog="MonitorSound.py")
    p.add_argument('--port', required=True, help='Serial port (COMx or /dev/ttyUSBx)')
    p.add_argument('--baud', type=int, default=921600)
    p.add_argument('--duration', type=float, default=10.0, help='seconds (<=0 for until Ctrl+C)')
    args = p.parse_args()
    
    try:
        ser = open_serial(args.port, args.baud)
        parse_stream(ser, 0)
    except Exception as inst:
        print(type(inst))    # the exception type
        print(inst.args)     # arguments stored in .args
        print(inst) 
        return

    

if __name__ == '__main__':
    main()
