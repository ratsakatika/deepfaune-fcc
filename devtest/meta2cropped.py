import os
import sys
import json
import contextlib
import pandas as pd
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
import urllib.request
import threading
curdir = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(curdir+'/../')
from load_api_results import load_api_results


class JSON2Cropped_Web:
    def __init__(self, jsonfilename, threshold, meta_data, chunk, chunk_size):
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            try:
                self.df_json, _ = load_api_results(jsonfilename)
                if 'failure' in self.df_json.keys():
                    self.df_json = self.df_json[self.df_json['failure'].isnull()]
                    self.df_json.reset_index(drop=True, inplace = True)
                    self.df_json.drop('failure', axis=1, inplace=True)
            except json.decoder.JSONDecodeError:
                self.df_json = []
        print("json loaded")
        self.df_json["basename"] = self.df_json.file.apply(lambda x: "".join(x.replace("\\", "/").split("/")[-2:]))
        meta_data["basename"] = meta_data.rawname.apply(lambda x: "".join(x.split("/")[-2:]))
        self.df_json = pd.merge(self.df_json, meta_data[["basename", "filename", "rawname"]], on="basename")
        print("df filtered, size: ", len(self.df_json))
        self.df_json = self.df_json.iloc[chunk_size*chunk: chunk_size*(chunk + 1)].reset_index(drop=True)
        print("df chunk, size: ", len(self.df_json))
        self.threshold = threshold
        self.k = 0
        self.kbox = 0
        self.imagecv = None
        self.preload_cache = {}
        self._lock = threading.Lock()
        self._threads = {}

    def nextBoxDetection(self):
        if self.k >= len(self.df_json):
            raise IndexError
        filename = self.df_json["filename"][self.k]
        conf = None
        if len(self.df_json['detections'][self.k]):
            if self.kbox == 0:
                self.nextImread()
            if self.df_json['detections'][self.k][self.kbox]['conf'] > self.threshold:
                category = int(self.df_json['detections'][self.k][self.kbox]['category'])
                croppedimage = self.cropCurrentBox()
                conf = self.df_json['detections'][self.k][self.kbox]['conf']
            else:
                category = 0
                croppedimage = None
            self.kbox += 1
            if self.kbox >= len(self.df_json['detections'][self.k]):
                self.k += 1
                self.kbox = 0
        else:
            category = 0
            croppedimage = None
            self.k += 1
            self.kbox = 0
        return croppedimage, category, filename, conf

    def convertJSONboxToBox(self):
        box_norm = self.df_json['detections'][self.k][self.kbox]["bbox"]
        height, width, _ = self.imagecv.shape
        xmin = int(box_norm[0] * width)
        ymin = int(box_norm[1] * height)
        xmax = xmin + int(box_norm[2] * width)
        ymax = ymin + int(box_norm[3] * height)
        return [xmin, ymin, xmax, ymax]

    def cropCurrentBox(self):
        if self.imagecv is None:
            return None, np.zeros(4)
        box = self.convertJSONboxToBox()
        croppedimage = cropSquareCVtoPIL(self.imagecv, box)
        return croppedimage, box

    def loadImage(self, filename):
        try:
            req = urllib.request.urlopen(filename.replace(" ", "%20"))
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            return cv2.imdecode(arr, -1)
        except Exception as e:
            print(f"[Error loading image] {filename}: {e}", file=sys.stderr)
            return None

    def preloadImagesAsync(self, indices):
        for idx in indices:
            if idx >= len(self.df_json) or idx in self.preload_cache or idx in self._threads:
                continue
            filename = str(self.df_json["rawname"][idx])
            def _load(index=idx, fname=filename):
                img = self.loadImage(fname)
                with self._lock:
                    self.preload_cache[index] = img
                    if index in self._threads:
                        del self._threads[index]
            thread = threading.Thread(target=_load, daemon=True)
            thread.start()
            self._threads[idx] = thread

    def nextImread(self):
        if self.k in self.preload_cache:
            with self._lock:
                self.imagecv = self.preload_cache[self.k]
                del self.preload_cache[self.k]
        else:
            filename = str(self.df_json["rawname"][self.k])
            self.imagecv = self.loadImage(filename)
        self.preloadImagesAsync([self.k + i for i in range(1, 200)])

def cropSquareCVtoPIL(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0,int(y1)):min(int(y2),height),max(0,int(x1)):min(int(x2),width)]
    croppedimage = Image.fromarray(croppedimagecv[:,:,(2,1,0)]) # converted to PIL BGR imagez
    return croppedimage


if __name__ == "__main__":
    import argparse
    from time import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=0, help="chunk id to process (should be between 0 and number_of_images // chunk_size)")
    parser.add_argument("--chunk_size", type=int, default=100000, help="number of images to process in each chunk")
    parser.add_argument("--meta_data", type=str, default='meta_data/lilaser.csv')
    parser.add_argument("--out_dir", type=str, default="cropped_data/")
    parser.add_argument("--json", type=str, default="snapshot-serengeti-2025-03-22-v1000.0.0-redwood_detections.filtered.json", help="json of MD detections")
    parser.add_argument("--thresh", type=float, default=0.5, help="detection confidence threshold")
    args = parser.parse_args()

    meta_data = pd.read_csv(args.meta_data)
    print("metadata loaded")
    detector = JSON2Cropped_Web(args.json, args.thresh, meta_data, args.chunk, args.chunk_size)

    print(args.chunk, "starting")
    j = 0
    t = time()
    nb_box = defaultdict(int)
    path_to_conf = {}
    while True and detector.k < len(detector.df_json):
        j += 1
        try:
            croppedimage, category, filename, conf = detector.nextBoxDetection()
            kbox = nb_box[filename]
            if croppedimage is not None:
                out = args.out_dir + "/".join(filename.split("/")[1:3]) + "/" + Path(filename).stem + f"_crop{kbox}.jpg"
                dirname = os.path.dirname(out)
                if not os.path.exists(dirname):
                    print(f"create folder {dirname}")
                    os.makedirs(dirname)
                croppedimage[0].save(out)
            nb_box[filename] += 1
            if conf is not None:
                path_to_conf[out] = conf
        except:
            print(args.chunk, "error", detector.k)
        if j%5000 == 0:
            print(time()-t, j, detector.k)
    print(args.chunk, "end", time()-t, j)
    pd.DataFrame(dict(cropname=path_to_conf.keys(), score=path_to_conf.values())).to_csv(f"conf_{args.chunk}.csv", index=False)
