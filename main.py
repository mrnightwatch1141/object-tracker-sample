'''
mr.nightwatch
11.03.2026

Dimostrazione improvvisata
gestione del traffico
con modello YOLO
'''
# Librerie
import cv2
import supervision as sv
import numpy as np
import threading
import time
import csv
from ultralytics import YOLO
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Variabili globali
percorso_video = None
esecuzione = False  # FIX: inizializzata a livello di modulo

'''
Funzione scegli_file()
Apre un filedialog per selezionare il video
'''
def scegli_file():
    global percorso_video
    percorso_video = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.mov")]
    )

    if percorso_video:
        file_lbl.config(text=f"Selezionato: {percorso_video}", fg="black")

'''
Funzione monitor_traffico()
Cattura il video, rileva i veicoli con YOLO e li traccia con ByteTrack
'''
def monitor_traffico():
    global esecuzione, conteggio_elementi, n_macchine, n_bus, n_camion, n_persone

    # Inizializzo la cattura video e il modello YOLO
    cattura = cv2.VideoCapture(percorso_video)
    modello = YOLO("yolov8n.pt")

    tracker = sv.ByteTrack(lost_track_buffer=60)

    # Set Classi di interesse (nomi COCO usati da YOLOv8)
    CLASSI_OGGETTI = {"car", "truck", "bus", "person"}

    # Conteggio veicoli
    conteggio_elementi = set()
    n_macchine        = 0
    n_bus             = 0
    n_camion          = 0
    n_persone         = 0

    while esecuzione:
        ret, frame = cattura.read()

        if not ret:
            break

        # Ridimensiona il frame
        display_width, display_height = 600, 350
        frame = cv2.resize(frame, (display_width, display_height))

        risultati = modello(frame, stream=True)

        for info in risultati:
            detections = sv.Detections.from_ultralytics(info)

            '''
            Filtra solo le classi oggetto con confidenza > 0.6:
            - Creo una maschera booleana
            - Lista dei valori di confidenza per ogni rilevazione
            - Accoppia ogni class_id con la sua confidenza corrispondente
            - Controlla se il nome della classe è nel set degli oggetti di interesse
            - Tiene solo le rilevazioni con confidenza superiore al 60%
            - Entrambe le condizioni devono essere VERE
            - Infine converte la List Comprehension in un array NumPy booleano

            Struttura List Comprehension:
            [espressione for variabile in iterabile]
            '''
            mask = np.array([
                modello.names[int(classe)] in CLASSI_OGGETTI and float(confidence) > 0.6
                # Unpacking della Tupla, quindi assegna ogni valore alla sua variabile
                # attraverso il metodo zip()
                for classe, confidence in zip(detections.class_id, detections.confidence)
            ], dtype=bool)
            # Mantiene solo le rilevazioni che rispettano il filtro
            detections = detections[mask]

            if len(detections) == 0:
                continue

            detections = tracker.update_with_detections(detections)

            # Disegna le box e traccia i veicoli
            for i in range(len(detections)):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                obj_id       = int(detections.tracker_id[i])
                tipo_oggetto = modello.names[int(detections.class_id[i])]

                # Se l'oggetto non è stato ancora contato, loggalo
                if obj_id not in conteggio_elementi:
                    conteggio_elementi.add(obj_id)

                    root.after(0, log_oggetto, obj_id, tipo_oggetto)

                    # Aggiorna i contatori in base al tipo
                    if tipo_oggetto == "car":
                        n_macchine += 1
                    elif tipo_oggetto == "bus":
                        n_bus += 1
                    elif tipo_oggetto == "truck":
                        n_camion += 1
                    elif tipo_oggetto == "person":
                        n_persone += 1

                # Disegna la box e l'ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {obj_id} {tipo_oggetto}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )

        try:
            frame_rgb =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img       =  Image.fromarray(frame_rgb)
            imgtk     =  ImageTk.PhotoImage(image=img)

            def aggiorna_GUI(i=imgtk, tot=len(conteggio_elementi), mac=n_macchine, bus=n_bus, cam=n_camion, pers=n_persone):
                video_lbl.imgtk = i
                video_lbl.configure(image=i)
                conteggio_elementi_var.set(tot)
                n_macchine_var.set(mac)
                n_bus_var.set(bus)
                n_camion_var.set(cam)
                n_persone_var.set(pers)

            root.after(0, aggiorna_GUI)

        except Exception as error:
            print(f"Errore aggiornamento GUI: {error}")
        # Limito a 60 FPS
        time.sleep(1 / 60)

    cattura.release()

'''
Funzione log_oggetto(obj_id, tipo_oggetto)
Aggiunge una riga alla Treeview dei log
'''
def log_oggetto(obj_id, tipo_oggetto):
    timestamp = time.strftime("%d-%m-%Y %H:%M:%S")

    # Evita doppioni con lo stesso ID del oggetto
    for child in log_list.get_children():
        if log_list.item(child)["values"][0] == obj_id:
            return

    log_list.insert("", "end", values=(obj_id, tipo_oggetto, timestamp))
    conteggio_elementi_var.set(len(conteggio_elementi))

'''
Funzione avvia_monitor()
'''
def avvia_monitor():
    global esecuzione
    if percorso_video:
        esecuzione = True
        threading.Thread(target=monitor_traffico, daemon=True).start()
    else:
        file_lbl.config(text="Inserire un video!", fg="red")

'''
Funzione stop_monitor()
'''
def stop_monitor():
    global esecuzione, conteggio_elementi, n_macchine, n_bus, n_camion, n_persone
    esecuzione = False
    # Reset contatori
    conteggio_elementi = set()
    n_macchine         = 0
    n_bus              = 0
    n_camion           = 0
    n_persone          = 0
    # Reset GUI
    conteggio_elementi_var.set(0)
    n_macchine_var.set(0)
    n_bus_var.set(0)
    n_camion_var.set(0)
    n_persone_var.set(0)

'''
Funzione scarica_logs()
'''
def scarica_logs():
    logs = []
    for child in log_list.get_children():
        logs.append(log_list.item(child)["values"])

    percorso_salvataggio = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV Files", "*.csv")]
    )
    if percorso_salvataggio:
        with open(percorso_salvataggio, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("ID Oggetto", "Tipo oggetto", "Timestamp"))
            writer.writerows(logs)

'''
GUI
'''
# Inizializza finestra Tkinter
root = Tk()
root.title("Rilevazione intelligente oggetti")
root.geometry("1024x600")
root.resizable(True, True)

# Configurazione stile
stile = ttk.Style()
stile.configure("Treeview.Heading", font=("Arial", 10, "bold"))
stile.configure("Treeview", font=("Arial", 10))

# Notebook per le schede
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Scheda 1: Monitoraggio video
scheda_monitor = Frame(notebook)
notebook.add(scheda_monitor, text="Monitoraggio")

# Scheda 2: Logs
scheda_log = Frame(notebook)
notebook.add(scheda_log, text="Logs")

# Frame di selezione video
file_frame = Frame(scheda_monitor)
file_frame.pack(fill="x", pady=5)

file_lbl = Label(file_frame, text="Nessun file selezionato", font=("Arial", 12))
file_lbl.pack(side="left", padx=10)
file_btn = Button(
    file_frame, text="Scegli file", command=scegli_file,
    font=("Arial", 12), bg="blue", fg="white"
)
file_btn.pack(side="right", padx=10)

# Frame Copyright
frame_copyright = Frame(scheda_monitor)
frame_copyright.pack(side="bottom", pady=5)
Label(frame_copyright, text="\xa9 2026 Angelo De Florio", font=("Arial", 10), fg="gray").pack()

# Bottoni Avvia e Ferma
frame_btn = Frame(scheda_monitor)
frame_btn.pack(side="bottom", pady=10)

avvia_btn = Button(
    frame_btn, text="Avvia", command=avvia_monitor,
    font=("Arial", 14), bg="green", fg="white"
)
avvia_btn.pack(side="left", padx=10)

stop_btn = Button(
    frame_btn, text="Ferma", command=stop_monitor,
    font=("Arial", 14), bg="red", fg="white"
)
stop_btn.pack(side="left", padx=10)

# Frame conteggio veicoli
frame_conteggio = Frame(scheda_monitor)
frame_conteggio.pack(side="bottom", pady=10)

# Inizializzazione contatori
conteggio_elementi_var = IntVar(value=0)
n_macchine_var         = IntVar(value=0)
n_bus_var              = IntVar(value=0)
n_camion_var           = IntVar(value=0)
n_persone_var          = IntVar(value=0)

Label(frame_conteggio, text="Oggetti Totali:", font=("Arial", 14)).pack(side="left", padx=5)
Label(frame_conteggio, textvariable=conteggio_elementi_var, font=("Arial", 14), fg="red").pack(side="left")

Label(frame_conteggio, text="Macchine:", font=("Arial", 14)).pack(side="left", padx=5)
Label(frame_conteggio, textvariable=n_macchine_var, font=("Arial", 14), fg="red").pack(side="left")

Label(frame_conteggio, text="Bus:", font=("Arial", 14)).pack(side="left", padx=5)
Label(frame_conteggio, textvariable=n_bus_var, font=("Arial", 14), fg="red").pack(side="left")

Label(frame_conteggio, text="Camion:", font=("Arial", 14)).pack(side="left", padx=5)
Label(frame_conteggio, textvariable=n_camion_var, font=("Arial", 14), fg="red").pack(side="left")

Label(frame_conteggio, text="Persone:", font=("Arial", 14)).pack(side="left", padx=5)
Label(frame_conteggio, textvariable=n_persone_var, font=("Arial", 14), fg="red").pack(side="left")

# Frame video
video_frame = Frame(scheda_monitor, bg="black")
video_frame.pack(fill="both", expand=True, padx=10, pady=10)

titolo_lbl = Label(
    video_frame, text="Rilevazione intelligente oggetti",
    font=("Arial", 18, "bold"), bg="black", fg="white"
)
titolo_lbl.pack(pady=5)

video_lbl = Label(video_frame, bg="black")
video_lbl.pack(fill="both", expand=True)

# Frame per le Log
frame_log_list = Frame(scheda_log)
frame_log_list.pack(padx=10, pady=10, fill="both", expand=True)

log_list = ttk.Treeview(
    frame_log_list,
    columns=("ID Oggetto", "Tipo di oggetto", "Timestamp"),
    show="headings"
)
log_list.heading("ID Oggetto",       text="ID Oggetto")
log_list.heading("Tipo di oggetto",  text="Tipo di oggetto")
log_list.heading("Timestamp",        text="Timestamp")
log_list.pack(fill="both", expand=True)

log_btns_frame = Frame(scheda_log)
log_btns_frame.pack(pady=10)

log_btn = Button(
    log_btns_frame, text="Scarica le logs", command=scarica_logs,
    font=("Arial", 12), bg="blue", fg="white"
)
log_btn.pack(side="left", padx=10)

root.mainloop()