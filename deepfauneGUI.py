# Copyright CNRS 2024

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import PySimpleGUI as sg
import cv2
import numpy as np
import threading
import io
import os
import torch
import multiprocessing
import urllib
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from hachoir.core import config
import subprocess
import tkinter as tk
from tkinter import ttk, TclError
import configparser
import base64
from b64_images import *
from contextlib import suppress
from PIL import Image, ImageDraw
from io import BytesIO
import platform
from datetime import datetime, date
import pandas as pd
from os import mkdir
from os.path import join, basename
from pathlib import Path
import pkgutil
import time
from collections import deque
from statistics import mean
import queue
import webbrowser
import copy
import shutil
import json
import sys

DFPATH = os.path.abspath(os.path.dirname(sys.argv[0]))
sys.path.append(DFPATH)
from predictTools import txt_undefined, txt_empty, txt_noanimalclasses
from detectTools import DFYOLO_NAME, MDSYOLO_NAME
from classifTools import txt_animalclasses

config.quiet = True
multiprocessing.freeze_support()
os.environ["PYTORCH_JIT"] = "0"
####################################################################################
### VERSION
####################################################################################
VERSION = "1.4.1"

####################################################################################
### PARAMETERS
####################################################################################
VIDEO = False # by default
threshold = threshold_default = 0.8
maxlag = maxlag_default = 10 # seconds
listlang = ['fr', 'en', 'it', 'de', 'es']
cpu_only = False
listdetector = ['DFbsMDS', 'DFMDS', 'MDR'] # DF + backstop MDS / DF + MDS ensemble / MDR
txt_listdevice = ["auto", "cpu", "cuda"]  # we can add mps device (for mac with M1 chip and above)

#######################
# CONFIGPARSER
# from settings.ini
config = configparser.ConfigParser()

def configget(option, defaultvalue):
    config.read(os.path.join(DFPATH,'settings.ini'))
    try:
        if defaultvalue  in ['True','False']:
            value = config.getboolean('General',option)
        else:
            value = config.get('General',option)
    except configparser.NoOptionError:
        value = defaultvalue == 'True' if defaultvalue  in ['True','False'] else defaultvalue
    return(value)
            
def configsetsave(option, value):
    config.set('General', option, value)
    with open(os.path.join(DFPATH,'settings.ini'), 'w') as inif:
        config.write(inif)

LANG = configget('language', 'fr')
countactivated = configget('count', 'False')
humanbluractivated = configget('humanblur', 'False')
birdclassificationactivated = configget('birdclassification', 'False')
detectorname = configget('detectorname', listdetector[0])
curr_device = configget("device", txt_listdevice[0]) if not cpu_only else "cpu"

# Checking update if authorized or if more than two months since last checking
checkupdate = configget('checkupdate', 'True')
lastcheckupdate = configget('lastcheckupdate', '2000-01-01')
today = date.today()
last = date.fromisoformat(lastcheckupdate)
if (today - last).days > 60:
    checkupdate = True
    configsetsave('checkupdate', 'True')
                  
####################################################################################
### GUI TEXT
####################################################################################
txt_other =  {'fr':"autre", 'en':"other", 'it':"altro", 'de':"andere Klasse", 'es':"otro"}
txt_browse = {'fr':"Choisir", 'en':"Select", 'it':"Scegliere", 'de':"Wählen", 'es':"Seleccionar"}
txt_incorrect = {'fr':"Dossier incorrect - aucun media trouvé", 'en':"Incorrect folder - no media found",
                 'it':"File scorretto - media non trovato", 'de':"Falscher Ordner - keine Medien gefunden",
                 'es':"Carpeta incorrecta - ningún medio encontrado"}
txt_confidence = {'fr':"Seuil de confiance", 'en':"Confidence threshold",
                  'it':"Livello minimo di affidabilita", 'de':"Konfidenzniveau",
                  'es':"Umbral de confianza"}
txt_sequencemaxlag = {'fr':"Durée maximum entre images consécutives (secondes)\npour qu'elles soient d'une même séquence",
                      'en':"Maximum duration between consecutive images\nfor them to be in the same sequence (seconds)",
                      'it':"Durata massima tra immagini consecutive\n in una sequenza (secondi)",
                      'de':"Maximale Dauer zwischen aufeinanderfolgenden Bildern\n in einer Sequenz (Sekunden)",
                      'es':"Duración máxima entre imágenes consecutivas\npara que estén en la misma secuencia (segundos)"}
txt_configrun = {'fr':"Configurer et lancer", 'en':"Configure & Run",
                 'it':"Configurare e inviare", 'de':"Konfigurieren und starten", 'es':"Configurar y ejecutar"}
txt_run = {'fr':"Lancer", 'en':"Run", 'it':"Inviare", 'de':"Starten", 'es':"Ejecutar"}
txt_paramframe = {'fr':"Paramètres", 'en':"Parameters", 'it':"Parametri", 'de':"Parameter", 'es':"Parámetros"}
txt_selectclasses = {'fr':"Désélectionner les classes animales absentes de votre zone d'étude",
                     'en':"Unselect animal classes not found in your study area",
                     'it':"Deselezionare le animale classi assenti della zona di studio",
                     'de':"Deaktivieren Sie Animal Klassen, die in Ihrem Studiengebiet fehlen",
                     'es':"Deseleccionar las clases animales ausentes en su zona de estudio"}
txt_birdclassification = {'fr':"Classification des oiseaux",
                      'en':"Bird Classification",
                      'it':"Classificazione degli uccelli",
                      'de':"Vogelklassifikation",
                      'es':"Clasificación de aves"}
txt_birdclassificationsingle = {'fr':"Classe unique",
                                'en':"Single class",
                                'it':"Classe unica",
                                'de':"Einzelne Klasse",
                                'es':"Clase única"}
txt_birdclassificationmultiple = {'fr':"Plusieurs classes",
                                 'en':"Multiple classes",
                                 'it':"Classi multiple",
                                 'de':"Mehrere Klassen",
                                 'es':"Varias clases (más lento)"}
txt_all = {'fr':"toutes", 'en':"all", 'it':"tutte", 'de':"Alles", 'es':"todas"}
txt_classnotfound = {'fr':"Aucun média pour cette classe", 'en':"No media found for this class",
                     'it':"Nessun media per questa classe", 'de':"Keine Medien für diese Klasse gefunden",
                     'es':"Ningún medio encontrado para esta clase"}
txt_filename = {'fr':"Nom de fichier", 'en':"Filename", 'it':"Nome del file", 'de':"Dateiname", 'es':"Nombre de archivo"}
txt_prediction = {'fr':"Prédiction", 'en':"Prediction", 'it':"Predizione", 'de':"Vorhersage", 'es':"Predicción"}
txt_seqnum = {'fr':"Séquence ID", 'en':"Sequence ID", 'it':"Sequenza ID", 'de':"Sequenz ID", 'es':"Secuencia ID"}
txt_error = {'fr':"Erreur", 'en':"Error", 'it':"Errore", 'de':"Fehler", 'es':"Error"}
txt_errorclass = {'fr':"erreur", 'en':"error", 'it':"errore", 'de':"Fehler", 'es':"error"}
txt_fileerror = {'fr':"Fichier illisible", 'en':"Unreadable file", 'it':"File illeggibile", 'de':"Unlesbare Datei", 'es':"Archivo ilegible"}
txt_savepredictions = {'fr':"Voulez-vous enregistrer les prédictions dans ", 'en':"Do you want to save predictions in ",
                       'it':"Volete registrare le predizioni nel ", 'de':"Möchten Sie Vorhersagen speichern",
                       'es':"¿Desea guardar las predicciones en "}
txt_destcopy = {'fr':"Copier dans des sous-dossiers de", 'en':"Copy in subfolders of",
                'it':"Copiare nei sotto file di", 'de':"In Unterordner Kopieren",
                'es':"Copiar en subcarpetas de"}
txt_destmove = {'fr':"Déplacer vers des sous-dossiers de", 'en':"Move to subfolders of",
                'it':"Spostare nei sotto file di", 'de':"In Unterordner Verschieben",
                'es':"Mover a subcarpetas de"}
txt_loadingmetadata = {'fr':"Chargement des metadonnées... (cela peut prendre du temps)", 'en':"Loading metadata... (this may take a while)",
                       'it':"Carica dei metadata... (puo essere lungo)", 'de':"Laden der Metadaten... (dies kann eine Weile dauern)",
                       'es':"Cargando metadatos... (esto puede tardar un poco)"}
txt_copyingfiles = {'fr':"Copie des fichiers... (cela peut prendre du temps)", 'en':"Copying files... (this may take a while)",
                    'it':"Copia di file... (puo essere lungo)", 'de':"Dateien kopieren... (dies kann eine Weile dauern)",
                    'es':"Copiando archivos... (esto puede tardar un poco)"}
txt_restart = {'fr':"Redémarrage nécessaire pour changer la langue. Arréter le logiciel ?",
               'en':"Restart required to change the language. Stopping the software?",
               'it':"Per cambiare la lingua è necessario un riavvio. Arresto del software ?",
               'de':"Neustart erforderlich, um die Sprache zu ändern. Wollen Sie die Software stoppen?",
               'es':"Es necesario reiniciar para cambiar el idioma. ¿Desea cerrar el software?"}
txt_visitwebsite = {'fr': "Aller sur le site", 'en': 'Visit the website',
                    'it': 'Vai al sito web', 'de': 'Auf die Website gehen', 'es': "Ir al sitio web"}
txt_newupdate = {'fr': "Mise à jour du logiciel", 'en': 'Software update',
                'it': 'Aggiornamento software', 'de': 'Software-Update', 'es': "Actualización de software"}
txt_newupdatelong = {'fr': "Une nouvelle mise à jour est disponible sur le site",
                     'en': 'A new update is available on the website',
                     'it': 'Un nuovo aggiornamento è disponibile sul sito web',
                     'de': 'Ein neues Update ist auf der Website verfügbar',
                     'es': "Una nueva actualización está disponible en el sitio web"}
txt_disablecheckupdate = {'fr': "Ne plus me le rappeler", 'en': "Do not remind me again",
                          'it': 'Non ricordarmelo più', 'de': 'Erinnere mich nicht mehr daran',
                          'es': "No recordármelo de nuevo"}
txt_enablecheckupdate = {'fr': "Me le rappeler plus tard", 'en': "Remind me later",
                         'it': 'Ricordamelo più tardi', 'de': 'Erinnere mich später',
                         'es': "Recuérdamelo más tarde"}
tooltip_metadata = {'fr': 'Metadata', 'en': 'Metadata', 'it': 'Metadati', 'de': 'Metadaten', 'es': 'Metadatos'}
tooltip_playpause = {'fr': 'Lire la vidéo/séquence', 'en': 'Play the video/sequence',
                     'it': 'Riproduci il video/sequenza', 'de': 'Video/Sequenz abspielen',
                     'es': "Reproducir el video/secuencia"}
tooltip_openfolder = {'fr': "Afficher le fichier dans Windows Explorer", 'en': 'Show file in Windows Explorer',
                      'it': 'Mostra il file in Windows Explorer', 'de': 'Datei im Windows Explorer anzeigen',
                      'es': "Mostrar archivo en Windows Explorer"}
tooltip_count = {'fr':"Comptage des animaux", 'en':"Animal count",
                 'it':"Conteggio degli animali", 'de':"Tiere zählung", 'es':"Conteo de animales"}
tooltip_counthuman = {'fr':"Comptage des humains", 'en':"Human count",
                      'it':"Conteggio degli esseri umani", 'de':"Zählung der Menschen", 'es':"Conteo de personas"}

####################################################################################
### THEME SETTINGS
####################################################################################
DEFAULT_THEME = {'accent': '#24a0ed', 'background': '#1c1c1c', 'text': '#d7d7d7', 'alt_background': '#2f2f2f'}
accent_color, text_color, background_color, alt_background = DEFAULT_THEME['accent'], DEFAULT_THEME['text'], DEFAULT_THEME['background'], DEFAULT_THEME['alt_background']

SUN_VALLEY_TCL = os.path.join(DFPATH,'theme/sun-valley.tcl')
SUN_VALLEY_THEME = 'dark' # 'light' not coherent with DEFAULT THEME
FONT_NORMAL = 'Segoe UI', 11
FONT_SMALL = 'Segoe UI', 10
FONT_LINK = 'Segoe UI', 11, 'underline'
FONT_TITLE = 'Segoe UI', 14
FONT_MED = 'Segoe UI', 12
FONT_TAB = 'Meiryo UI', 10
LINK_COLOR = '#3ea6ff'


####################################################################################
### GUI UTILS
####################################################################################
def debugprint(txt_fr, txt_en, end='\n'):
    if LANG=="fr":
        print(txt_fr, end=end)
    else:
        print(txt_en, end=end)

def draw_boxes(imagecv, box=None):
    if box is not None:
        if np.count_nonzero(box)>0: # is not default empty box
            cv2.rectangle(imagecv, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), max(1,imagecv.shape[0]//200))

def blur_boxes(imagecv, boxes=None):
    if boxes is not None:
        for box in boxes:
            if np.count_nonzero(box)>0: # is not default empty box
                ROI = imagecv[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                blur = cv2.blur(ROI, (151,151)) 
                imagecv[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = blur


def get_radio_device(curr_device, txt_listdevice):
    radio_device = [("☑ " if curr_device == device else "☐ ") + device for device in txt_listdevice]
    if not torch.cuda.is_available():
        radio_device[2] = "!" + radio_device[2]
    return radio_device


def copyfile_blur(src, dst, boxes=None):
    if boxes is None:
        shutil.copyfile(src, dst)
    else:
        imagecv = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        blur_boxes(imagecv, boxes)
        cv2.imwrite(dst, imagecv)

def dialog_get_dir(title, initialdir=None):
    # rooting to the main PySimpleGUI window
    # does not work here dute to color problems in the dialog box
    root = tk.Tk()
    root.tk.call('source', SUN_VALLEY_TCL)
    root.tk.call('set_theme', 'light')
    root.withdraw()
    selectdir = tk.filedialog.askdirectory(title=title, initialdir=initialdir, parent=root)
    if len(selectdir) == 0:
        selectdir = None
    root.destroy()
    return selectdir

def dialog_get_file(title, initialdir, initialfile, defaultextension):
    # rooting to the main PySimpleGUI window
    # does not work here dute to color problems in the dialog box
    root = tk.Tk()
    root.tk.call('source', SUN_VALLEY_TCL)
    root.tk.call('set_theme', 'light')
    root.withdraw()
    selectfile = tk.filedialog.asksaveasfilename(initialdir=initialdir, initialfile=initialfile, defaultextension=defaultextension, parent=root)
    if len(selectfile) == 0:
        selectfile = None
    root.destroy()
    return selectfile

def dialog_yesno(message):
    # rooting to the main PySimpleGUI window
    yesorno = tk.messagebox.askquestion('', message, icon='warning', parent=window.TKroot)
    return yesorno

def dialog_error(message):
    # rooting to the main PySimpleGUI window
    tk.messagebox.showerror(title=txt_error[LANG], message=message, parent=window.TKroot)
    
def popup(message):
    layout = [[sg.Text(message, background_color=background_color, text_color=text_color)]]
    windowpopup = sg.Window('Message', layout, no_titlebar=True, keep_on_top=True,
                            font = FONT_MED, background_color=background_color, finalize=True)
    with suppress(tk.TclError):
        windowpopup.TKroot.tk.call('source', SUN_VALLEY_TCL)
    windowpopup.TKroot.tk.call('set_theme', SUN_VALLEY_THEME)
    return windowpopup

def scrollabled_text_window(text, title):
    root = tk.Tk()
    root.tk.call('source', SUN_VALLEY_TCL)
    root.tk.call('set_theme', 'dark')
    root.title(title)
    scrollbar = tk.Scrollbar(root)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget = tk.Text(root, wrap=tk.WORD, yscrollcommand=scrollbar.set, width=50, height=20)
    text_widget.pack(expand=True, fill='both')
    scrollbar.config(command=text_widget.yview)
    text_widget.insert(tk.END, text)
    text_widget.config(state=tk.DISABLED)
    root.mainloop()

def StyledButton(button_text, fill, text_color, background_color, font=None, tooltip=None, key=None, visible=True,
              pad=None, bind_return_key=False, button_width=None):
    multi = 4
    btn_w = ((len(button_text) if button_width is None else button_width) * 5 + 20) * multi
    height = 18 * multi
    btn_img = Image.new('RGBA', (btn_w, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(btn_img)
    x0 = y0 = 0
    radius = 10 * multi
    d.ellipse((x0, y0, x0 + radius * 2, height), fill=fill)
    d.ellipse((btn_w - radius * 2 - 1, y0, btn_w - 1, height), fill=fill)
    d.rectangle((x0 + radius, y0, btn_w - radius, height), fill=fill)
    data = io.BytesIO()
    btn_img.thumbnail((btn_w // 3, height // 3))
    btn_img.save(data, format='png', quality=100)
    btn_img = base64.b64encode(data.getvalue())
    return sg.Button(button_text=button_text, image_data=btn_img,
                     button_color=(text_color, background_color), mouseover_colors=(text_color, background_color),
                     tooltip=tooltip, key=key, pad=pad, enable_events=True, size=(button_width, 1),
                     bind_return_key=bind_return_key, font=font, visible=visible, border_width=0)

def StyledMenu(menu_definition, text_color, background_color, text_font, key):    
    bar_text = text_color
    bar_bg = background_color
    bar_font = text_font
    font = text_font
    menu_bg = background_color
    menu_text = text_color
    disabled_text_color = 'gray'
    row = []
    for menu in menu_def:
        text = menu[0]
        if sg.MENU_SHORTCUT_CHARACTER in text:
            text = text.replace(sg.MENU_SHORTCUT_CHARACTER, '')
        if text.startswith(sg.MENU_DISABLED_CHARACTER):
            disabled = True
            text = text[len(sg.MENU_DISABLED_CHARACTER):]
        else:
            disabled = False
        button_menu = sg.ButtonMenu(text, menu, border_width=0, button_color=(bar_text, bar_bg), key=text, pad=(0, 0), disabled=disabled,
                                    font=bar_font, item_font=font, disabled_text_color=disabled_text_color, text_color=menu_text, background_color=menu_bg) #, tearoff=tearoff)
        button_menu.part_of_custom_menubar = True
        #button_menu.custom_menubar_key = key if key is not None else k
        row += [button_menu]
    return(sg.Column([row], pad=(0,0), background_color=bar_bg, expand_x=True, key=key))

####################################################################################
### CHECKING SCREEN SIZE & RESOLUTION FOR IMAGE DISPLAY
####################################################################################
# Image display
def cv2bytes(imagecv, imsize=None):
    if imsize is not None and imsize[0]>0 and imsize[1]>0:
        imagecv_resized = cv2.resize(imagecv, imsize)
    else:
        imagecv_resized = imagecv
    is_success, png_buffer = cv2.imencode(".png", imagecv_resized)
    bio = BytesIO(png_buffer)
    return bio.getvalue()

# Initial logo image
startimagecv = cv2.imdecode(np.fromfile(os.path.join(DFPATH,'icons/startscreen-large.png'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)    
curimagecv = startimagecv

# Checking screen possibilities and sizing image accordinglyimport ctypes
DEFAULTIMGSIZE = (width,height) = (933,700)
try:
    if platform.platform().lower().startswith("windows"):
        root = sg.tk.Tk()
        root.attributes("-alpha", 0)
        root.state('zoomed')
        root.update()
        width  = root.winfo_width()
        height = root.winfo_height()
        root.destroy()
    else:
       width, height = sg.Window.get_screen_size() 
except:
    pass


correctedimgsize = (min(DEFAULTIMGSIZE[0],int(width*0.65)),
                    min(DEFAULTIMGSIZE[1],int(width*0.65*DEFAULTIMGSIZE[0]/DEFAULTIMGSIZE[1]), int(height*0.75)))

curimagecv = cv2.resize(curimagecv, correctedimgsize)

####################################################################################
### MAIN GUI WINDOW
####################################################################################
sorted_txt_classes_lang = []
sorted_txt_animalclasses_lang = sorted(txt_animalclasses[LANG])

# Main window
txt_file = {'fr':"Fichier", 'en':"File",
            'it':"File", 'de':"Datei", 'es':"Archivo"}
txt_pref = {'fr':"Préférences", 'en':"Preferences",
            'it':"Preferenze", 'de':"Präferenzen", 'es':"Preferencias"}
txt_help = {'fr':"Aide", 'en':"Help",
            'it':"Aiuto", 'de':"Hilfe", 'es':"Ayuda"}
txt_helponline = {'fr':"Aide en ligne", 
                'en':"Online help",
                'it':"Aiuto in linea", 
                'de':"Online-Hilfe", 
                'es':"Ayuda en línea"}
txt_import = {'fr':"Importer", 'en':"Import",
              'it':"Caricare", 'de':"Importieren", 'es':"Importar"}
txt_importimage = {'fr':"Images", 'en':"Images",
                   'it':"Immagine", 'de':"Bilder", 'es':"Imágenes"}
txt_importvideo = {'fr':"Vidéos", 'en':"Videos",
                   'it':"Video", 'de':"Videos", 'es':"Videos"}
txt_export = {'fr':"Exporter les résultats", 'en':"Export results",
              'it':"Esportare i risultati", 'de':"Resultate exportieren", 'es':"Exportar resultados"}
txt_ascsv = {'fr':"Format CSV", 'en':"As CSV",
             'it':"Formato CSV", 'de':"Als CSV", 'es':"Formato CSV"}
txt_asxlsx = {'fr':"Format XSLX", 'en':"As XSLX",
              'it':"Formato XSLX", 'de':"Als XLSX", 'es':"Formato XLSX"}
txt_createsubfolders = {'fr':"Créer des sous-dossiers", 'en':"Create subfolders",
                        'it':"Creare dei sotto file", 'de':"Unterordner erstellen", 'es':"Crear subcarpetas"}
txt_copy = {'fr':"Copier les fichiers", 'en':"Copy files",
            'it':"Copiare i file", 'de':"Dateien kopieren", 'es':"Copiar archivos"}
txt_copywithhumanblur = {'fr':"Copier les fichiers (avec floutage des humains)",
                         'en':"Copy files (with human blurring)",
                         'it':"Copiare i file (con la sfocatura degli umani)",
                         'de':"Dateien kopieren (mit Die Unschärfe von Menschen)",
                         'es':"Copiar archivos (con difuminado de personas)"}
txt_language = {'fr':"Langue", 'en':"Language",
                'it':"Lingua", 'de':"Sprache", 'es':"Idioma"}
txt_activatecount = {'fr':"Activer le comptage (expérimental)", 'en':"Activate count (experimental)",
                     'it':"Attivare il conto (sperimentale)", 'de':"Zählung aktivieren (experimentell)",
                     'es':"Activar el conteo (experimental)"}
txt_deactivatecount = {'fr':"Désactiver le comptage (expérimental)", 'en':"Deactivate count (experimental)",
                       'it':"Disattivare il conto (sperimentale)", 'de':"Zählung desaktivieren (experimentell)",
                       'es':"Desactivar el conteo (experimental)"}
txt_activatehumanblur = {'fr':"Activer le floutage des humains (images seulement)", 'en':"Activate human blurring (image only)",
                         'it':"Attivare la sfocatura degli umani (solo immagini)", 'de':"Die Unschärfe von Menschen aktivieren (nur die Bilder)",
                         'es':"Activar el difuminado de personas (solo imágenes)"}
txt_deactivatehumanblur = {'fr':"Desactiver le floutage des humains  (images seulement)", 'en':"Deactivate human blurring  (image only)",
                           'it':"Disattivare la sfocatura degli umani (solo immagini)", 'de':"die Unschärfe von Menschen desaktivieren (nur die Bilder)",
                           'es':"Desactivar el difuminado de personas (solo imágenes)"}
txt_models = {'fr': "Modèles", 'en': 'Models',
              'it': 'Modelli', 'de': 'Modelle', 'es': "Modelos"}
txt_detector = {'fr': "Détecteur", 'en': 'Detector',
                'it': 'Rivelatore', 'de': 'Detektor', 'es': "Detector"}
txt_device = {"fr": "Device", "en": "Device",
              "it": "Device", "de": "Device", "es": "Device"}
txt_detectordesc = {
    'fr': ["Performant et rapide", "Plus performant mais plus lent", "Excellent mais beaucoup plus lent"],
    'en': ["Efficient and fast", "More efficient but slower", "Excellent but much slower"],
    'it': ["Performante e veloce", "Più performante ma più lento", "Eccellente ma molto più lento"],
    'de': ["Leistungsfähig und schnell", "Leistungsstärker, aber langsamer", "Ausgezeichnet, aber deutlich langsamer"],
    'es': ["Eficiente y rápido", "Más eficiente pero más lento", "Excelente pero mucho más lento"]
}
txt_credits = {'fr':"A propos de DeepFaune", 
               'en':"About DeepFaune",
               'it':"Informazioni su DeepFaune", 
               'de':"Über DeepFaune", 
               'es':"Acerca de DeepFaune"}
txt_listdetector = [("☑ " if detectorname == listdetector[0] else "☐ ") + txt_detectordesc[LANG][0], 
                    ("☑ " if detectorname == listdetector[1] else "☐ ") + txt_detectordesc[LANG][1], 
                    ("☑ " if detectorname == listdetector[2] else "☐ ") + txt_detectordesc[LANG][2]]

if countactivated:
    txt_statuscount = txt_deactivatecount[LANG]
else:
    txt_statuscount = txt_activatecount[LANG]
if humanbluractivated:
    txt_statushumanblur = txt_deactivatehumanblur[LANG]
    txt_subfoldersoptions = [txt_copy[LANG], txt_copywithhumanblur[LANG]]
else:
    txt_statushumanblur = txt_activatehumanblur[LANG]
    txt_subfoldersoptions = [txt_copy[LANG]]
radio_device = get_radio_device(curr_device, txt_listdevice)

menu_def = [
    ['&'+txt_file[LANG], [
        '&'+txt_import[LANG],[txt_importimage[LANG],txt_importvideo[LANG]],
        '!'+txt_export[LANG],[txt_ascsv[LANG],txt_asxlsx[LANG]],
        '!'+txt_createsubfolders[LANG], txt_subfoldersoptions
    ]],
    ['&'+txt_pref[LANG], [
        txt_language[LANG], listlang,
        txt_statuscount,
        '!'+txt_statushumanblur] + 
        ([] if cpu_only else [txt_device[LANG], radio_device])
    ],
    ['&'+txt_models[LANG], [
        txt_detector[LANG], txt_listdetector
    ]],
    ['&'+txt_help[LANG], [
        'Version', [VERSION],
        txt_helponline[LANG],
        txt_credits[LANG]
    ]]
]

# Constants for slider design
SLIDER_WIDTH = 32
SLIDER_HEIGHT = 120
SLIDER_HANDLE_RADIUS = 9

# Function to draw the vertical slider on the Graph
def draw_slider(graph, value, enabled):
    graph.erase()
    handle_y = value * SLIDER_HEIGHT
    handle_y = min(handle_y, SLIDER_HEIGHT - SLIDER_HANDLE_RADIUS)
    handle_y = max(handle_y, SLIDER_HANDLE_RADIUS)
    trough_x = SLIDER_WIDTH / 2
    graph.draw_line((trough_x, 0), (trough_x, SLIDER_HEIGHT), color='#cccccc', width=4)
    if enabled:
        graph.draw_circle((trough_x, handle_y), SLIDER_HANDLE_RADIUS, fill_color=accent_color, line_color=accent_color)
    else:
        graph.draw_circle((trough_x, handle_y), SLIDER_HANDLE_RADIUS, fill_color='#cccccc', line_color='#cccccc')

# On windows, there is a button to open the selected file in explorer
if platform.platform().lower().startswith("windows"):
    button_openfile = [sg.Button(image_data=OPEN_FOLDER_ICON, key="-OPENFILE-", button_color=(background_color,background_color), border_width=0,
                                 tooltip=tooltip_openfolder[LANG])]
else:
    button_openfile = []

# MAIN LAYOUT
layout = [
    [
        StyledMenu(menu_def, text_color=text_color, background_color=background_color, text_font=FONT_NORMAL, key='-MENUBAR-')
    ],
    [
        [sg.Frame('',[
            [
                sg.Column([
                    [sg.Table(values=[[]], font=FONT_NORMAL,
                              headings=[txt_filename[LANG]], justification = "l", 
                              vertical_scroll_only=False, auto_size_columns=False, col_widths=[20], expand_y=True,
                              enable_events=True, select_mode = sg.TABLE_SELECT_MODE_BROWSE,
                              background_color=background_color, text_color=text_color,
                              key='-TAB-')],
                    [
                        sg.Combo(values=[txt_all[LANG]]+sorted_txt_classes_lang+[txt_undefined[LANG],txt_empty[LANG]],
                                 background_color=background_color, text_color=text_color, enable_events=True,
                                 default_value=txt_all[LANG], size=(17, 1), bind_return_key=False, key='-RESTRICT-'),
                        sg.Button(key='-PREVIOUS-', image_data=PREVIOUS_BUTTON_IMG, button_color=(background_color,background_color), border_width=0, tooltip=None),
                        sg.Button(key='-NEXT-', image_data=NEXT_BUTTON_IMG, button_color=(background_color,background_color), border_width=0, tooltip=None)
                     ]
                ], background_color=background_color, expand_y=True),
                sg.Column([ 
                    [sg.Frame('',
                              [[sg.Image(cv2bytes(curimagecv), key='-IMAGE-',  background_color=background_color)]]
                              , background_color=background_color, border_width=0)
                     ],
                    [sg.Text(txt_prediction[LANG]+':', background_color=background_color, text_color=text_color, size=(10, 1)),
                     sg.Combo(values=list(sorted_txt_classes_lang+[txt_undefined[LANG],txt_other[LANG],txt_empty[LANG]]),
                              default_value="", enable_events=True,
                              background_color=background_color, text_color=text_color, size=(20, 1), bind_return_key=True, key='-PREDICTION-'),
                     sg.Text("   Score: 0.0", background_color=background_color, text_color=text_color, key='-SCORE-'),
                     sg.Text("", background_color=background_color, text_color=text_color, key='-SEQNUM-'),
                     sg.Text(" ", background_color=background_color, text_color=text_color),
                     sg.Image(ANIMAL_ICON, background_color=background_color, visible=countactivated, key='-COUNT-', tooltip=tooltip_count[LANG]),
                     sg.Text(":", background_color=background_color, text_color=text_color, visible=countactivated, key='-COUNTCOLON-'),
                     sg.Input(default_text="0", size=(5, 1), enable_events=True, key='-COUNTER-', background_color=background_color, text_color=text_color, visible=countactivated,
                              disabled_readonly_background_color=background_color, disabled_readonly_text_color=text_color, border_width=0),
                     sg.Image(HUMAN_ICON, background_color=background_color, visible=countactivated, key='-COUNTHUMAN-', tooltip=tooltip_counthuman[LANG]),
                     sg.Text(":", background_color=background_color, text_color=text_color, visible=countactivated, key='-COUNTHUMANCOLON-'),
                     sg.Input(default_text="0", size=(5, 1), enable_events=True, key='-COUNTERHUMAN-', background_color=background_color, text_color=text_color, visible=countactivated,
                              disabled_readonly_background_color=background_color, disabled_readonly_text_color=text_color, border_width=0)]
                ], background_color=background_color, expand_x=True),
                sg.Column([
                    [sg.Image(BRIGHTNESS_ICON, background_color=background_color)],
                    [sg.Graph(
                         canvas_size=(SLIDER_WIDTH, SLIDER_HEIGHT),
                         graph_bottom_left=(0, 0),
                         graph_top_right=(SLIDER_WIDTH, SLIDER_HEIGHT),
                         key='-GAMMALEVEL-',
                         enable_events=True,
                         drag_submits=True,  # Enable drag events
                         background_color=background_color,
                     )],
                    [sg.HorizontalSeparator(pad=10)],
                    [sg.Button(key='-PLAY-', image_data=NICE_PLAYIN_ICON, button_color=(background_color,background_color), border_width=0, enable_events=True, tooltip=tooltip_playpause[LANG])],
                    button_openfile,
                    [sg.Button(key='-METADATA-', image_data=INFO_ICON, button_color=(background_color,background_color), tooltip=tooltip_metadata[LANG], border_width=0)],
                ], background_color=background_color, expand_y=True)
            ]
        ], background_color=background_color, expand_y=True, border_width=0)]
    ],
    [
        sg.Frame('',[
            [
                StyledButton(txt_configrun[LANG], accent_color, "gray", background_color, key='-CONFIGRUN-', button_width=8+len(txt_configrun[LANG]), pad=(5, (7, 5))),
                sg.ProgressBar(1, orientation='h', border_width=1, expand_x=True, key='-PROGBAR-', bar_color=accent_color), sg.Text("00:00:00", background_color=background_color, text_color=text_color, key='-RTIME-')
            ],
        ], expand_x=True, background_color=background_color, border_width=0)
    ]
]

window = sg.Window("DeepFaune - CNRS",layout, margins=(0,0),
                   font = FONT_MED, location=(0, 0),
                   resizable=True, background_color=background_color).Finalize()
window.read(timeout=0)
window['-PREDICTION-'].Update(disabled=True)
window['-RESTRICT-'].Update(disabled=True)
window['-COUNTER-'].Update(disabled=True)
window['-COUNTER-'].bind("<Return>", "_Enter") # to generate an event only after return key
window['-COUNTER-'].bind("<KP_Enter>", "_Enter") # to generate an event only after return key in numeric pad (on Linux)
window['-COUNTERHUMAN-'].Update(disabled=True)
window['-COUNTERHUMAN-'].bind("<Return>", "_Enter") # to generate an event only after return key
window['-COUNTERHUMAN-'].bind("<KP_Enter>", "_Enter") # to generate an event only after return key in numeric pad (on Linux)
window.bind('<Configure>', '-CONFIG-') # to generate an event when window is resized
window['-IMAGE-'].bind('<Double-Button-1>' , "DOUBLECLICK-")

with suppress(TclError):
    window.TKroot.tk.call('source', SUN_VALLEY_TCL)
window.TKroot.tk.call('set_theme', SUN_VALLEY_THEME) # if dark, implies -CONFIG- events due to internal additionnal padding

####################################################################################
### GUI UTILS (after it is created)
####################################################################################
def updateMenuImport(disabled):
    if disabled == True:
        menu_def[0][1][0] = '!'+txt_import[LANG]
    else:
        menu_def[0][1][0] = '&'+txt_import[LANG]
    window[txt_file[LANG]].Update(menu_def[0])

def updateMenuExport(disabled):
    if disabled == True:
        menu_def[0][1][2] = '!'+txt_export[LANG]
    else:
        menu_def[0][1][2] = '&'+txt_export[LANG]
    window[txt_file[LANG]].Update(menu_def[0])

def updateMenuSubfolders(disabled):
    if disabled == True:
        menu_def[0][1][4] = '!'+txt_createsubfolders[LANG]
    else:
        menu_def[0][1][4] = '&'+txt_createsubfolders[LANG]
    window[txt_file[LANG]].Update(menu_def[0])

def updateMenuCount(activated):
    if activated == True:
        menu_def[1][1][2] = txt_deactivatecount[LANG]
    else:
        menu_def[1][1][2] = txt_activatecount[LANG]
    window[txt_pref[LANG]].Update(menu_def[1])
    
def updateMenuHumanBlur(activated):
    if activated == True:
        if not VIDEO:
            menu_def[1][1][3] = txt_deactivatehumanblur[LANG]
            menu_def[0][1][5] = [txt_copy[LANG], txt_copywithhumanblur[LANG]]
        else:
            menu_def[1][1][3] = '!'+txt_deactivatehumanblur[LANG]
            menu_def[0][1][5] = [txt_copy[LANG]]
            
    else:
        if not VIDEO:
            menu_def[1][1][3] = txt_activatehumanblur[LANG]
        else:
            menu_def[1][1][3] = '!'+txt_activatehumanblur[LANG]
        menu_def[0][1][5] = [txt_copy[LANG]]
    window[txt_pref[LANG]].Update(menu_def[1])
    window[txt_file[LANG]].Update(menu_def[0])


def updateMenuDetector(detectorname):
    global txt_listdetector
    txt_listdetector = [("☑ " if detectorname == listdetector[0] else "☐ ") + txt_detectordesc[LANG][0],
                        ("☑ " if detectorname == listdetector[1] else "☐ ") + txt_detectordesc[LANG][1],
                        ("☑ " if detectorname == listdetector[2] else "☐ ") + txt_detectordesc[LANG][2]]
    menu_def[2][1][1] = txt_listdetector
    window[txt_models[LANG]].Update(menu_def[2])


def updateMenuDevice(device):
    global radio_device
    radio_device = get_radio_device(device, txt_listdevice)
    menu_def[1][1][5] = radio_device
    window[txt_pref[LANG]].Update(menu_def[1])


def updatePredictionInfo(disabled):
    if disabled is True:
        window['-PREDICTION-'].Update(value="")
        window['-PREDICTION-'].Update(disabled=True)
        window['-SCORE-'].Update("   Score: 0.0")
        if countactivated:
            window['-COUNTER-'].Update(value=0)
            window['-COUNTER-'].Update(disabled=True)
            window['-COUNTERHUMAN-'].Update(value=0)
            window['-COUNTERHUMAN-'].Update(disabled=True)
        if VIDEO:
            window['-SEQNUM-'].Update("")
        else:
            window['-SEQNUM-'].Update(" "+txt_seqnum[LANG]+": NA")
    else:
        window['-PREDICTION-'].Update(disabled=False)
        if countactivated:
            window['-COUNTER-'].Update(disabled=False)
            window['-COUNTERHUMAN-'].Update(disabled=False)

def updateTxtNewClasses(txt_newclass):
    if txt_newclass not in sorted_txt_classes_lang+[txt_undefined[LANG],txt_other[LANG],txt_empty[LANG]]+txt_new_classes_lang:
        txt_new_classes_lang.append(txt_newclass) 
        window['-PREDICTION-'].Update(values=sorted(sorted_txt_classes_lang+txt_new_classes_lang)+[txt_undefined[LANG],txt_other[LANG],txt_empty[LANG]],
                                      value=txt_newclass)
        valuerestrict = values['-RESTRICT-']
        window['-RESTRICT-'].Update(values=[txt_all[LANG]]+sorted(sorted_txt_classes_lang+txt_new_classes_lang)+[txt_undefined[LANG],txt_empty[LANG]],
                                    value=valuerestrict)

def resetTxtClasses(txt_classes_lang_new):
    sorted_txt_classes_lang = sorted(txt_classes_lang_new)
    window['-PREDICTION-'].Update(values=sorted_txt_classes_lang+[txt_undefined[LANG],txt_other[LANG],txt_empty[LANG]],
                                      value="")
    window['-RESTRICT-'].Update(values=[txt_all[LANG]]+sorted_txt_classes_lang+[txt_undefined[LANG],txt_empty[LANG]],
                                value=txt_all[LANG])
    return sorted_txt_classes_lang
 
def rescale_slider(value, min_rescale=-10, max_rescale=10):
    return gamma_dict[int((1-value)*min_rescale + value*max_rescale)]


def update_slider(value=None, enabled=None):
    global slider_value, slider_enabled, is_value_updated
    is_value_updated = True
    if value is not None:
        if value is not None and slider_value is not None and rescale_slider(value) == rescale_slider(slider_value):
            is_value_updated = False
        slider_value = value
    if enabled is not None:
        slider_enabled = enabled
    draw_slider(slider_graph, slider_value, slider_enabled)


def gamma_correction(imagecv, gamma):
    if abs(gamma-1.0)<1e-6:
        return imagecv
    # Build a lookup table mapping pixel values [0, 255] to their gamma-corrected values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(imagecv, table)

imageOffset = (0,0) # space between the window and the image control itself
def updateImage(newcurimagecv=None, gamma=1.):
    global curimagecv
    if newcurimagecv is not None:
        curimagecv = newcurimagecv
    curimsize = ((window.size[0] - imageOffset[0], window.size[1] - imageOffset[1]))
    window['-IMAGE-'].update(data=cv2bytes(gamma_correction(curimagecv, gamma), curimsize))

def resizeImage():
    global curwindowsize
    if window.size[0] != curwindowsize[0] or window.size[1] != curwindowsize[1]:
        updateImage()
    curwindowsize = window.size

####################################################################################
### GUI IN ACTION
####################################################################################

#########################
## GLOBAL VARIABLES
#########################
## GUI's variables
curridx = -1 # current filenames index
rowidx = -1 # current tab row index
subsetidx = [] # current subset of filenames
testdir = None
thread = None
thread_queue = queue.Queue()
predictorready = False
txt_new_classes_lang = []

## misc variables to allow resizing
configactive = False # checks if a series of config events is in progress
nbconfigseries = 0 # nb of series of config events
curwindowsize = (0,0) # current size before config events

# slider variable
gamma_dict = {k: -k/10*0.2 + (1+k/10) if k <= 0 else 3*k/10 + (1-k/10) for k in range(-10, 11)}
slider_graph = window['-GAMMALEVEL-']
slider_enabled = None
slider_value = None
is_value_updated = True
update_slider(0.5, False)


#########################
## ASYNCHRONOUS ACTIONS
#########################
def runPredictor(): # predictor in action in a separate thread
    batchduration = deque(maxlen=20)
    if VIDEO:
        while True:
            start = time.time()
            batch, k1, k2 = predictor.nextBatch()
            end = time.time()
            if k1==nbfiles: # last batch done
                break
            batchduration.append(end-start)
            rtime = time.strftime("%H:%M:%S", time.gmtime(mean(batchduration)*(nbfiles-batch)))
            progbar = batch/nbfiles
            thread_queue.put([rtime, progbar, k1, k2])
    else:
        while True:
            start = time.time()
            batch, k1, k2, k1seq, k2seq = predictor.nextBatch()
            end = time.time()
            if k1==nbfiles:  # last batch done
                break
            batchduration.append(end-start)
            rtime = time.strftime("%H:%M:%S", time.gmtime(mean(batchduration)*(1+int(nbfiles/BATCH_SIZE)-batch)))
            progbar = batch*BATCH_SIZE/nbfiles
            thread_queue.put([rtime, progbar, k1seq, k2seq])
    thread_queue.put(["00:00:00", 1.0, nbfiles, nbfiles])

def updateFromThreadQueue(): # updating GUI using info in thread queue
    global thread, thread_queue
    try:
        rtime, progbar, k1, k2 = thread_queue.get(0)
        window['-RTIME-'].Update(rtime)
        window['-PROGBAR-'].update_bar(progbar)
        window['-TAB-'].Update(row_colors=tuple((k,accent_color,background_color)
                                                for k in range(k1, k2)))
        if curridx>=k1 and curridx<k2: # current media must be refreshed
            rowidx = values['-TAB-'][0]
            # touching position in Table, will send a -TAB- event
            window['-TAB-'].update(select_rows=[rowidx])
    except queue.Empty:
        pass
    if thread is not None:
        ## enabling GUI events when thread has terminated
        if thread.is_alive() == False:
            thread = None
            updateMenuImport(disabled=False)
            window['-RESTRICT-'].Update(disabled=False)
            if countactivated:
                window['-COUNTER-'].Update(disabled=False)
                window['-COUNTERHUMAN-'].Update(disabled=False)
            updatePredictionInfo(disabled=False)
            updateMenuExport(disabled=False)
            updateMenuSubfolders(disabled=False)
            window['-CONFIGRUN-'].Update(button_color=(background_color, background_color))

def playVideoUntilOtherEvent(k):
    global curimagecv
    previmagecv = curimagecv  
    videocap = cv2.VideoCapture(filenames[k])
    total_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames==0:
        framecv = None # corrupted video, considered as empty
        event, values = window.read(timeout=10)
    else:
        play = True
        kframe = 0
        while(play):
            event, values = window.read(timeout=10)
            if slider_enabled and event == '-GAMMALEVEL-':
                _, mouse_y = values['-GAMMALEVEL-']
                if 0 <= mouse_y <= SLIDER_HEIGHT:
                    update_slider(round((mouse_y / SLIDER_HEIGHT), 2), None)
            
            videocap.set(cv2.CAP_PROP_POS_FRAMES, kframe)
            ret, framecv = videocap.read()
            if ret==True: # uncorrupted frame
                updateImage(framecv, rescale_slider(slider_value))
            kframe = kframe+5
            if kframe>=total_frames:
                kframe = 0
            updateFromThreadQueue()
            if event != '__TIMEOUT__':
                if event == "-PLAY-": # pause button
                    play = False
                elif event != '-CONFIG-' and event != "-GAMMALEVEL-" and event != "-GAMMALEVEL-+UP":
                    play = False
                
    videocap.release()
    # updating position in Table,
    # such that we focus on the row/file of interest,
    # but will send a -TAB- event
    #if event != '-TAB-':
    #    rowidx = values['-TAB-'][0]
    #    window['-TAB-'].update(select_rows=[rowidx])
    #    window['-TAB-'].Widget.see(rowidx+1)
    updateImage(previmagecv, rescale_slider(slider_value))
    window['-PLAY-'].Update(image_data=NICE_PLAYIN_ICON)
    return event, values

def playSequenceUntilOtherEvent(filename):
    global curimagecv, humanbluractivated
    previmagecv = curimagecv
    play = True
    nbfiles = len(filenames)
    k1 = k2 = curridx
    while predictor.getSeqnums()[curridx]==predictor.getSeqnums()[max(0,k1-1)] and k1>0:
        k1 = k1-1
    while predictor.getSeqnums()[curridx]==predictor.getSeqnums()[min(nbfiles-1,k2+1)] and k2<(nbfiles-1):
        k2 = k2+1
    k = k1
    if k1==k2: # singleton
        event, values = window.read(timeout=10)
        play = False
        window['-PLAY-'].Update(image_data=NICE_PLAYIN_ICON)
        return event, values
    t0 = time.time()
    while(play):
        event, values = window.read(timeout=10)
        if slider_enabled and event == '-GAMMALEVEL-':
            _, mouse_y = values['-GAMMALEVEL-']
            if 0 <= mouse_y <= SLIDER_HEIGHT:
                update_slider(round((mouse_y / SLIDER_HEIGHT), 2), None)
        try:
            imagecv = cv2.imdecode(np.fromfile(filenames[k], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except:
            imagecv = np.zeros((DEFAULTIMGSIZE[1],DEFAULTIMGSIZE[0],3), np.uint8)
        if humanbluractivated:
            blur_boxes(imagecv, predictor.getHumanBoxes(filenames[k]))
        updateImage(imagecv, rescale_slider(slider_value))
        if time.time() - t0 > 0.5:
            k = k + 1
            t0 = time.time()
        if k>k2:
            k = k1
        updateFromThreadQueue()
        if event != '__TIMEOUT__':
            if event == "-PLAY-": # pause button
                play = False
            elif event != '-CONFIG-' and event != "-GAMMALEVEL-" and event != "-GAMMALEVEL-+UP":
                play = False
    updateImage(previmagecv, rescale_slider(slider_value))
    window['-PLAY-'].Update(image_data=NICE_PLAYIN_ICON)
    return event, values
   
#########################
## MAIN LOOP
#########################
DEBUG = False

draw_popup_update = False
draw_meta = False
try:
    online_version = urllib.request.urlopen('https://pbil.univ-lyon1.fr/software/download/deepfaune/.version', timeout=1)
    online_version = online_version.read().decode().replace("\n", "")
except:
    online_version = None

if checkupdate and online_version:
    configsetsave('lastcheckupdate',  date.today().isoformat())
    v_online_parts = list(map(int, online_version.split('.')))
    v_installed_parts = list(map(int, VERSION.split('.')))
    for v_online, v_installed in zip(v_online_parts, v_installed_parts):
        if v_online > v_installed:
            draw_popup_update = True

while True:
    event, values = window.read(timeout=10)
    if event != "__TIMEOUT__" and DEBUG is True:
        print(event)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    
    # If the user clicks or drags the slider
    if slider_enabled and event == '-GAMMALEVEL-':  # Detect initial click
        _, mouse_y = values['-GAMMALEVEL-']
        if 0 <= mouse_y <= SLIDER_HEIGHT:
            update_slider(round((mouse_y / SLIDER_HEIGHT), 2), None)

    #########################
    ## PLAYING VIDEO ?
    #########################
    if (event == '-IMAGE-DOUBLECLICK-' or event == '-PLAY-') and VIDEO and (len(subsetidx)>0):
        window['-PLAY-'].Update(image_data=NICE_PAUSE_ICON)
        event, values = playVideoUntilOtherEvent(curridx) # captures the window event internally
    #########################
    ## PLAYING SEQUENCE ?
    #########################
    if (event == '-IMAGE-DOUBLECLICK-' or event == '-PLAY-') and not VIDEO and (len(subsetidx)>0) and predictorready:
        window['-PLAY-'].Update(image_data=NICE_PAUSE_ICON)
        event, values = playSequenceUntilOtherEvent(filenames[curridx]) # captures the window event internally
        
    #########################
    ## WINDOW RESIZING ?
    #########################
    if event == '-CONFIG-': # respond to window resize event
        configactive = True
    elif event != '-CONFIG-' and configactive == True:
        nbconfigseries = nbconfigseries+1
        configactive = False
        if nbconfigseries>1: # the first config events are internal at starting time, not a resizing event
            resizeImage()
        else:
            curwindowsize = window.size # current size before other config events (resizing or moving)
            imageOffset = (window.size[0] - window['-IMAGE-'].get_size()[0],
                           window.size[1] - window['-IMAGE-'].get_size()[1]) # offset is set after the the first config events
    #########################
    ## CHECK UPDATE
    #########################
    if draw_popup_update:
        layoutupdate = [
            [sg.Text(txt_newupdatelong[LANG] + f" (version {online_version})", expand_x=True, background_color=background_color, text_color=text_color)], 
            [StyledButton(txt_visitwebsite[LANG], accent_color, background_color, background_color,
                          button_width=15+len(txt_visitwebsite[LANG]), key='-UPDATE-'),
            StyledButton(txt_enablecheckupdate[LANG], accent_color, background_color, background_color,
                         button_width=15+len(txt_enablecheckupdate[LANG]), key='-UPDATECHECK-'),
            StyledButton(txt_disablecheckupdate[LANG], accent_color, background_color, background_color,
                         button_width=15+len(txt_disablecheckupdate[LANG]), key='-NOUPDATECHECK-')]]

        windowupdate = sg.Window(txt_newupdate[LANG], layoutupdate,  
                                 font = FONT_MED, margins=(0, 0),
                                 background_color=background_color, finalize=True)
        with suppress(TclError):
            windowupdate.TKroot.tk.call('source', SUN_VALLEY_TCL)
        windowupdate.TKroot.tk.call('set_theme', SUN_VALLEY_THEME) # if dark, implies -CONFIG- events due to internal additionnal padding

        while draw_popup_update:
            eventconfig, valuesconfig = windowupdate.read(timeout=10)
            if eventconfig == '-UPDATE-':
                webbrowser.open("https://www.deepfaune.cnrs.fr")
                draw_popup_update = False
            if eventconfig == '-NOUPDATECHECK-':
                configsetsave('checkupdate', 'False')
                draw_popup_update = False
            elif eventconfig in (sg.WIN_CLOSED, 'Exit', '-UPDATECHECK-'):
                draw_popup_update = False
        windowupdate.close()
        window.TKroot.focus_force()
    if event in listlang:
        #########################
        ## SELECTING LANGUAGE
        #########################
        configsetsave('language', event)
        if event != LANG:
            # reinit fordibdden classes
            configsetsave('forbiddenanimalclasses', '[]')
            # restarting required
            yesorno = dialog_yesno(txt_restart[LANG])
            if yesorno == 'yes':
                break
    elif event in txt_listdetector:
        #########################
        ## SELECTING DETECTOR
        #########################
        if event == txt_listdetector[0]:
            detectorname = listdetector[0]
        if event == txt_listdetector[1]:
            detectorname = listdetector[1]
        if event == txt_listdetector[2]:
            detectorname = listdetector[2]
        updateMenuDetector(detectorname)
        configsetsave("detectorname", detectorname)
    elif event == txt_helponline[LANG]:        
        webbrowser.open("https://deepfaune.pages.math.cnrs.fr/software/")
        continue
    elif event in radio_device:
        #########################
        ## SELECTING DEVICE
        #########################
        curr_device = txt_listdevice[radio_device.index(event)]
        updateMenuDevice(curr_device)
        configsetsave("device", curr_device)

    elif event == txt_activatecount[LANG]:
        #########################
        ## (DE)ACTIVATING COUNT
        #########################
        countactivated = True
        if predictorready and len(subsetidx)>0:
            _, _, _, count_curridx = predictor.getPredictions(curridx)
            window['-COUNTER-'].Update(value=count_curridx)
            humancount_curridx =  predictor.getHumanCount(curridx)
            window['-COUNTERHUMAN-'].Update(value=humancount_curridx)
        else:
            window['-COUNTER-'].Update(value=0)
            window['-COUNTERHUMAN-'].Update(value=0)
        window['-COUNT-'].Update(visible=True)
        window['-COUNTCOLON-'].Update(visible=True)
        window['-COUNTER-'].Update(visible=True)
        window['-COUNTHUMAN-'].Update(visible=True)
        window['-COUNTHUMANCOLON-'].Update(visible=True)
        window['-COUNTERHUMAN-'].Update(visible=True)
        if predictorready and thread==None:
            window['-COUNTER-'].Update(disabled=False)
            window['-COUNTERHUMAN-'].Update(disabled=False)
        configsetsave('count', 'True')
        updateMenuCount(activated=True)
    elif event == txt_deactivatecount[LANG]:
        countactivated = False
        window['-COUNT-'].Update(visible=False)
        window['-COUNTCOLON-'].Update(visible=False)
        window['-COUNTER-'].Update(visible=False)
        window['-COUNTHUMAN-'].Update(visible=False)
        window['-COUNTHUMANCOLON-'].Update(visible=False)
        window['-COUNTERHUMAN-'].Update(visible=False)
        configsetsave('count', 'False')
        updateMenuCount(activated=False)
    elif event == txt_activatehumanblur[LANG]:
        #########################
        ## (DE)ACTIVATING HUMAN BLUR
        #########################
        humanbluractivated = True
        configsetsave('humanblur', 'True')
        updateMenuHumanBlur(activated=True)
        # refresh current view; touching position in Table, will send a -TAB- event
        if testdir is not None:
            window['-TAB-'].update(select_rows=[rowidx])
    elif event == txt_deactivatehumanblur[LANG]:
        humanbluractivated = False
        configsetsave('humanblur', 'False')
        updateMenuHumanBlur(activated=False)
        # refresh current view; touching position in Table, will send a -TAB- event
        if testdir is not None:
            window['-TAB-'].update(select_rows=[rowidx])
    elif event == txt_credits[LANG]:
        #########################
        ## CREDITS
        #########################
        webbrowser.open("https://www.deepfaune.cnrs.fr")
        continue
    elif event == txt_importimage[LANG] or event == txt_importvideo[LANG]: 
        #########################
        ## LOADING MEDIAS
        #########################
        newtestdir = dialog_get_dir(txt_browse[LANG]) # we keep the previous testdir, if not None
        if newtestdir != None:
            testdir = newtestdir
            if event == txt_importimage[LANG]:
                VIDEO = False
                BATCH_SIZE = 8
            if event == txt_importvideo[LANG]:
                VIDEO = True
                BATCH_SIZE = 12
            predictorready = False
            curridx = -1
            window['-RTIME-'].Update("00:00:00")
            window['-PROGBAR-'].update_bar(0)
            updateImage(startimagecv)
            window['-RESTRICT-'].Update(value=txt_all[LANG], disabled=True)
            updatePredictionInfo(disabled=True)
            updateMenuExport(disabled=True)
            updateMenuSubfolders(disabled=True)
            updateMenuHumanBlur(activated=humanbluractivated)
            debugprint("Dossier sélectionné : "+testdir, "Selected folder: "+testdir)
            ### GENERATOR
            if VIDEO:
                filenames = sorted(
                    [str(f) for f in  Path(testdir).rglob('*.[Aa][Vv][Ii]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Mm][Pp]4') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Mm][Pp][Ee][Gg]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Mm][Oo][Vv]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Mm]4[Vv]') if not f.parents[1].match('*deepfaune_*')]
                )
            else:
                filenames = sorted(
                    [str(f) for f in  Path(testdir).rglob('*.[Jj][Pp][Gg]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Jj][Pp][Ee][Gg]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Bb][Mm][Pp]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Tt][Ii][Ff]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Gg][Ii][Ff]') if not f.parents[1].match('*deepfaune_*')] +
                    [str(f) for f in  Path(testdir).rglob('*.[Pp][Nn][Gg]') if not f.parents[1].match('*deepfaune_*')]
                )
            nbfiles = len(filenames)
            if VIDEO:
                debugprint("Nombre de vidéos : "+str(nbfiles), "Number of videos: "+str(nbfiles))
            else:
                debugprint("Nombre d'images : "+str(nbfiles), "Number of images: "+str(nbfiles))
            if nbfiles==0:
                testdir = None
                window['-TAB-'].Update(values=[[]])
                window['-CONFIGRUN-'].Update(button_color=("gray", background_color))
                update_slider(0.5, False)
                dialog_error(txt_incorrect[LANG])
            else:
                curridx = 0
                rowidx = 0
                subsetidx = list(range(0,len(filenames)))
                window['-TAB-'].Update(values=[[basename(f)] for f in filenames])
                window['-TAB-'].Update(row_colors=tuple((k,text_color,background_color)
                                                        for k in range(0, 1))) # bug, first row color need to be hard reset
                window['-TAB-'].update(select_rows=[0])
                window['-CONFIGRUN-'].Update(button_color=(background_color, background_color))
                update_slider(enabled=True)
    elif event == '-CONFIGRUN-' and testdir is not None and thread is None:
        #########################
        ## CONFIGURE
        #########################
        forbiddenanimalclasses_json = configget('forbiddenanimalclasses', '[]')
        forbiddenanimalclasses = json.loads(forbiddenanimalclasses_json)
        listCB = [] # Default selected animal classes
        lineCB = []
        for k in range(0,len(sorted_txt_animalclasses_lang)):
            lineCB = lineCB+[sg.CB(sorted_txt_animalclasses_lang[k],
                                   key=sorted_txt_animalclasses_lang[k], 
                                   default=(sorted_txt_animalclasses_lang[k] not in forbiddenanimalclasses),
                                   size=(14,1), background_color=background_color, text_color=text_color)]
            if k%4==3:
                listCB = listCB+[lineCB]
                lineCB = []
        if lineCB:
            listCB = listCB+[lineCB]
        select_frame = sg.Frame(txt_selectclasses[LANG], listCB, font=FONT_NORMAL, expand_x=True, expand_y=True,
                                background_color=background_color, border_width=0) # required here to avoid element reuse (not accepted) 
        if VIDEO:
            sequencespin = []
        else:
            sequencespin = [sg.Text(txt_sequencemaxlag[LANG]+'\t', expand_x=True, background_color=background_color, text_color=text_color),
                            sg.Spin(values=[i for i in range(0, 60)], initial_value=maxlag_default, size=(4, 1), enable_events=True, key='-LAG-', background_color=background_color, text_color=text_color)]
        layoutconfig = [
            [select_frame],
            [sg.Frame(txt_birdclassification[LANG], [[sg.Radio(txt_birdclassificationsingle[LANG], group_id=0, key='-BIRDCLASSIFICATIONSINGLE-', default=(birdclassificationactivated==False), size=(14,1), background_color=background_color, text_color=text_color),
                                     sg.Radio(txt_birdclassificationmultiple[LANG], group_id=0, key='-BIRDCLASSIFICATIONMULTIPLE-', default=(birdclassificationactivated==True), size=(25,1), background_color=background_color, text_color=text_color)]],
                      font=FONT_NORMAL, expand_x=True, expand_y=True,
                      background_color=background_color, border_width=0)],
            [sg.Frame(txt_paramframe[LANG], font=FONT_MED, expand_x=True, expand_y=True, pad=(sg.DEFAULT_ELEMENT_PADDING[1],10), #DEFAULT_ELEMENT_PADDING between elements (row, col)
                      layout=[
                          [sg.Text(txt_confidence[LANG]+'\t', expand_x=True, background_color=background_color, text_color=text_color),
                           sg.Spin(values=[i/100. for i in range(25, 100)], initial_value=threshold_default, size=(4, 1), enable_events=True,
                                   background_color=background_color, text_color=text_color, key='-THRESHOLD-')],
                          sequencespin
                      ], background_color=background_color, border_width=0)],
            [
                StyledButton(txt_run[LANG], accent_color, background_color, background_color, button_width=8+len(txt_run[LANG]), key='-RUN-')
            ]
        ]
        windowconfig = sg.Window(txt_configrun[LANG], copy.deepcopy(layoutconfig),  
                                 font = FONT_MED, margins=(0, 0),
                                 background_color=background_color, finalize=True)
        with suppress(TclError):
            windowconfig.TKroot.tk.call('source', SUN_VALLEY_TCL)
        windowconfig.TKroot.tk.call('set_theme', SUN_VALLEY_THEME) # if dark, implies -CONFIG- events due to internal additionnal padding

        configabort = False
        while True:
            eventconfig, valuesconfig = windowconfig.read(timeout=10)
            if eventconfig == '-RUN-':
                break
            elif eventconfig in (sg.WIN_CLOSED, 'Exit'):
                configabort = True
                break
        windowconfig.close()
        if not configabort:
            threshold = float(valuesconfig['-THRESHOLD-'])
            if not VIDEO:
                maxlag = float(valuesconfig['-LAG-'])
            forbiddenanimalclasses = []
            for label in sorted_txt_animalclasses_lang:
                if not valuesconfig[label]:
                    forbiddenanimalclasses += [label]
            if len(forbiddenanimalclasses):
                debugprint("Classes non selectionnées : ", "Unselected classes: ", end="")
                print(forbiddenanimalclasses)
            # Serialize to a JSON string and set it in config
            forbiddenanimalclasses_json = json.dumps(forbiddenanimalclasses)
            configsetsave('forbiddenanimalclasses', forbiddenanimalclasses_json)
        ########################
        ## RUN
        ########################
        if not configabort:
            window['-CONFIGRUN-'].Update(button_color=("gray", background_color))
            updateMenuImport(disabled=True)
            window['-RTIME-'].Update("00:00:00")
            window['-PROGBAR-'].update_bar(0)
            window['-RESTRICT-'].Update(value=txt_all[LANG], disabled=True)
            updatePredictionInfo(disabled=True)
            updateMenuExport(disabled=True)
            updateMenuSubfolders(disabled=True)
            birdclassificationactivated = valuesconfig['-BIRDCLASSIFICATIONMULTIPLE-']
            print(birdclassificationactivated)
            configsetsave('birdclassification', str(birdclassificationactivated))
            if VIDEO:
                from predictTools import PredictorVideo
            else:
                from predictTools import PredictorImage
            if VIDEO:
                predictor = PredictorVideo(filenames, threshold, LANG, birdclassificationactivated, BATCH_SIZE, detectorname=detectorname, device=curr_device)
            else:
                if len(filenames)>1000:
                    popup_win = popup(txt_loadingmetadata[LANG])
                predictor = PredictorImage(filenames, threshold, maxlag, LANG, birdclassificationactivated, BATCH_SIZE, detectorname=detectorname, device=curr_device)
                if len(filenames)>1000:
                    popup_win.close()
                seqnums = predictor.getSeqnums()
            sorted_txt_classes_lang = resetTxtClasses(predictor.getTxtClasses())
            filenames = predictor.getFilenames()
            curridx = 0
            rowidx = 0
            subsetidx = list(range(0,len(filenames)))
            window.Element('-TAB-').Update(values=[[basename(f)] for f in filenames]) # color reset is induced
            window['-TAB-'].update(select_rows=[0])
            window['-TAB-'].Update(row_colors=tuple((k,text_color,background_color)
                                                    for k in range(0, 1))) # bug, first row color need to be hard reset
            predictor.setForbiddenAnimalClasses(forbiddenanimalclasses)
            thread = threading.Thread(target=runPredictor)
            thread.daemon = True
            thread.start() 
            predictorready = True
    elif event == txt_ascsv[LANG] or event == txt_asxlsx[LANG]:
        #########################
        ## EXPORTING RESULTS
        #########################
        predictedclass, predictedscore, _, count = predictor.getPredictions()
        predictedtop1 = predictor.getPredictedTop1()
        if VIDEO:
            predictedclass_base, predictedscore_base = predictedclass, predictedscore
        else:
            predictedclass_base, predictedscore_base, _, _ = predictor.getPredictionsBase()
        if countactivated:
            preddf  = pd.DataFrame({'filename':predictor.getFilenames(), 'date':predictor.getDates(), 'seqnum':predictor.getSeqnums(),
                                    'predictionbase':predictedclass_base, 'scorebase':predictedscore_base,
                                    'prediction':predictedclass, 'score':predictedscore, 'top1':predictedtop1,
                                    'count':count, 'humancount':predictor.getHumanCount()})
        else:
            preddf  = pd.DataFrame({'filename':predictor.getFilenames(), 'date':predictor.getDates(), 'seqnum':predictor.getSeqnums(),
                                    'predictionbase':predictedclass_base, 'scorebase':predictedscore_base,
                                    'prediction':predictedclass, 'score':predictedscore, 'top1':predictedtop1,
                                    'humancount':predictor.getHumanCount()})
        preddf.sort_values(['seqnum','filename'], inplace=True)
        if event == txt_ascsv[LANG]:
            csvpath =  dialog_get_file(txt_savepredictions[LANG], initialdir=testdir, initialfile="deepfaune.csv", defaultextension=".csv")
            if csvpath:
                debugprint("Enregistrement dans "+csvpath, "Saving to "+csvpath)
                preddf.to_csv(csvpath, index=False)
        if event == txt_asxlsx[LANG]:
            xlsxpath =  dialog_get_file(txt_savepredictions[LANG], initialdir=testdir, initialfile="deepfaune.xlsx", defaultextension=".xlsx")
            if xlsxpath:
                debugprint("Enregistrement dans "+xlsxpath, "Saving to "+xlsxpath)
                preddf.to_excel(xlsxpath, index=False)
    elif event == "-OPENFILE-" and len(values['-TAB-'])>0:
        #########################
        ## FILE OPENING
        #########################
        if platform.platform().lower().startswith("windows"):
            subprocess.Popen(r'explorer /select, "' + filenames[curridx] + '"')
    elif event == '-METADATA-' and len(values['-TAB-'])>0:
        #########################
        ## SHOW METADATA
        #########################
        metadata = None
        try:
            parser = createParser(filenames[curridx])
        except:
            parser = None
        if parser:
            with parser:
                try:
                    metadata = extractMetadata(parser)
                except Exception as err:
                    pass
        if metadata:
            text = "\n".join(metadata.exportPlaintext()[1:])
            text = f"- Path: {filenames[curridx]}\n" + text
            scrollabled_text_window(text, "Metadata")

    elif (testdir is not None) \
         and (event == '-TAB-' and len(values['-TAB-'])>0) \
         and (len(subsetidx)>0) or (event == "-GAMMALEVEL-" and slider_enabled):
        #########################
        ## SHOW SELECTED MEDIA
        ## AND ITS PREDICTION
        #########################
        if event != "-GAMMALEVEL-" and slider_value != 0.5 and event != '-IMAGE-DOUBLECLICK-' and event != '-PLAY-':
            update_slider(0.5)
        rowidx = values['-TAB-'][0]       
        curridx = subsetidx[rowidx]
        if VIDEO:
            videocap = cv2.VideoCapture(filenames[curridx])
            total_frames = int(videocap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames==0:
                 imagecv = None # corrupted video, considered as empty
            else:
                if predictorready:
                    kframe = predictor.getKeyFrames(curridx) # possibly 0 if video not treated by predictor yet
                else:
                    kframe = 0
                videocap.set(cv2.CAP_PROP_POS_FRAMES, kframe)
                ret, imagecv = videocap.read()
                while ret==False and (kframe+lag)<=((BATCH_SIZE-1)*lag): # ignoring corrupted frames (useless when key frame are found by predictor)
                    kframe = kframe+lag
                    videocap.set(cv2.CAP_PROP_POS_FRAMES, kframe)
                    ret, imagecv = videocap.read()
                if ret==False:
                    imagecv = None                        
            videocap.release()
        else:
            try:
                imagecv = cv2.imdecode(np.fromfile(filenames[curridx], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            except:
                imagecv = None
        if imagecv is None:
            imagecv = np.zeros((DEFAULTIMGSIZE[1],DEFAULTIMGSIZE[0],3), np.uint8)
            cv2.putText(imagecv, text=txt_fileerror[LANG], org=(300, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(0, 0, 255),thickness=1)
            if predictorready:
                predictor.setPredictedClass(curridx, txt_errorclass[LANG], 0.0)
                window['-PREDICTION-'].update(value=txt_errorclass[LANG])
                window['-SCORE-'].Update("   Score: 0.0")
                if countactivated:
                    window['-COUNTER-'].Update(value=0)
                    window['-COUNTERHUMAN-'].Update(value=0)
        else:
            if predictorready:
                predictedclass_curridx, predictedscore_curridx, predictedbox_curridx, count_curridx = predictor.getPredictions(curridx)
                humancount_curridx = predictor.getHumanCount(curridx)
                txt_human = txt_noanimalclasses[LANG][0]
                if predictedclass_curridx != txt_human:
                    window['-PREDICTION-'].update(value=predictedclass_curridx)
                    if countactivated:
                        window['-COUNTER-'].Update(value=str(count_curridx))
                        window['-COUNTERHUMAN-'].Update(value=str(humancount_curridx))
                else:
                    window['-PREDICTION-'].update(value=txt_human)
                    if countactivated:
                        window['-COUNTER-'].Update(value=0) # setting the animal count to 0 when sequences is predicted as human
                        window['-COUNTERHUMAN-'].Update(value=str(humancount_curridx))
                window['-SCORE-'].Update("   Score: "+str(predictedscore_curridx))
                if humanbluractivated:
                    if not VIDEO:
                        blur_boxes(imagecv, predictor.getHumanBoxes(filenames[curridx]))
                if predictedclass_curridx is not txt_empty[LANG]:
                    draw_boxes(imagecv, predictedbox_curridx)
        if is_value_updated or event == "-TAB-":
            updateImage(imagecv, rescale_slider(slider_value))
        if predictorready and not VIDEO:
            window['-SEQNUM-'].Update(" "+txt_seqnum[LANG]+": "+str(seqnums[curridx]))
    elif (testdir is not None) \
         and (event == '-PREVIOUS-' or event == '-NEXT-') \
         and (len(subsetidx)>0):
        #########################
        ## NEXT/PREVIOUS MEDIA
        #########################
        rowidx = values['-TAB-'][0]
        if event == '-NEXT-':
            rowidx = rowidx+1
            if rowidx==len(subsetidx):
                rowidx = 0
        if event == '-PREVIOUS-':
            rowidx = rowidx-1
            if rowidx==-1:
                rowidx = len(subsetidx)-1
        curridx = subsetidx[rowidx]
        # updating position in Table, will send an event
        window['-TAB-'].update(select_rows=[rowidx])
        window['-TAB-'].Widget.see(rowidx+1)
    elif event == txt_copy[LANG] or event == txt_copywithhumanblur[LANG]:
        #########################
        ## CREATING SUBFOLDERS
        #########################
        def unique_new_filename(testdir, now, classname, basename):
            folder = join(join(testdir, "deepfaune_"+now, classname))
            if os.path.exists(join(folder, basename)):
                i = 2
                part1 = basename[:-4]
                part2 = basename[-4:]
                basename = f"{part1}_{i}{part2}"
                while os.path.exists(join(folder, basename)):
                    i += 1
                    basename = f"{part1}_{i}{part2}"
            return join(folder, basename)

        now = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        destdir = None
        if event == txt_copy[LANG] or event == txt_copywithhumanblur[LANG]:
            destdir = dialog_get_dir(txt_destcopy[LANG], initialdir=testdir)
            if destdir is not None:
                debugprint("Copie vers "+join(destdir,"deepfaune_"+now), "Copying to "+join(destdir,"deepfaune_"+now))
        if destdir is not None:
            if len(filenames)>500:
                popup_win = popup(txt_copyingfiles[LANG])
            predictedclass, predictedscore, _, _ = predictor.getPredictions()
            mkdir(join(destdir,"deepfaune_"+now))
            for subfolder in set(predictedclass):
                mkdir(join(destdir,"deepfaune_"+now,subfolder))
            if event == txt_copy[LANG]:
                for k in range(nbfiles):
                    shutil.copyfile(filenames[k], unique_new_filename(destdir, now, predictedclass[k], basename(filenames[k])))
            if event == txt_copywithhumanblur[LANG] and not VIDEO:
                for k in range(nbfiles):
                    copyfile_blur(filenames[k], unique_new_filename(destdir, now, predictedclass[k], basename(filenames[k])),
                                  predictor.getHumanBoxes(filenames[k]))
            if len(filenames)>500:
                popup_win.close()
    elif event == '-PREDICTION-':
        #########################
        ## CORRECTING PREDICTION
        #########################
        if predictorready:
            if VIDEO:
                predictor.setPredictedClass(curridx, values['-PREDICTION-'])
            else:
                predictor.setPredictedClassInSequence(curridx, values['-PREDICTION-'])
            window['-PREDICTION-'].Update(select=False)
            window['-SCORE-'].Update("   Score: 1.0")
        # new class proposed by the user ?
        updateTxtNewClasses(values['-PREDICTION-'])
    elif event == '-COUNTER-' + "_Enter":
        if predictorready:
            try:
                newcount = int(values['-COUNTER-'])
                predictor.setPredictedCount(curridx, newcount)
            except ValueError:
                count_curridx =  predictor.getPredictedCount(curridx)
                window['-COUNTER-'].Update(value=count_curridx)
    elif event == '-COUNTERHUMAN-' + "_Enter":
        if predictorready:
            try:
                newhumancount = int(values['-COUNTERHUMAN-'])
                predictor.setHumanCount(curridx, newhumancount)
            except ValueError:
                humancount_curridx =  predictor.getHumanCount(curridx)
                window['-COUNTERHUMAN-'].Update(value=humancount_curridx)
    elif event == '-RESTRICT-':
        #########################
        ## BROWSING RESTRICTION
        #########################
        if values['-RESTRICT-'] == txt_all[LANG]:
            subsetidx = list(range(0,len(filenames)))
        else:
            predictedclass, _, _, _ = predictor.getPredictions()
            txt_human = txt_noanimalclasses[LANG][0]
            if  values['-RESTRICT-'] != txt_human:
                # restricting to a species
                subsetidx = list(np.where(np.array(predictedclass)==values['-RESTRICT-'])[0])
            else:
                # restricting to human, whatever the other species present; human presence is checked
                subsetidx = list(np.where(np.array(predictor.getHumanPresence())==True)[0])
        if len(subsetidx)>0:
            updatePredictionInfo(disabled=False)
            update_slider(enabled=True)
            window.Element('-TAB-').Update(values=[[basename(f)] for f in [filenames[k] for k in subsetidx]])
            window['-TAB-'].Update(row_colors = tuple((k,accent_color,background_color)
                                                      for k in range(0, len(subsetidx)))) # row in accent_color because prediction is available
            window['-TAB-'].update(select_rows=[0])
        else:
            updatePredictionInfo(disabled=True)
            window.Element('-TAB-').Update(values=[[]])
            update_slider(0.5, False)
            updateImage(startimagecv)
            dialog_error(txt_classnotfound[LANG])
        curridx = 0
        rowidx = 0
    elif event == sg.TIMEOUT_KEY:
        window.refresh()
    #########################
    ## UPDATING GUI FROM THREAD INFO (thread-safe)
    #########################
    updateFromThreadQueue()
window.close()

