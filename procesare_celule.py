import os
import cv2
import numpy as np
folder_date = "/home/iernestrk/Downloads/DataSets/OncologieIOCN/MDA231_ELK-KD_R1-P1_E2_03.10.23_P6_10x"
def proceseaza_imagini():
    if not os.path.exists(folder_date):
        print(f"Nu exista folderul ala: {folder_date}")
        return
    fisiere = sorted([f for f in os.listdir(folder_date) if f.endswith('.jpg')])
    print(f"Am gasit {len(fisiere)} imagini. Incep treaba...")
    toate_traiectoriile = []
    for img_name in fisiere:
        cale_img = os.path.join(folder_date, img_name)
        img = cv2.imread(cale_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        blurred = cv2.medianBlur(img, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        _, thresh = cv2.threshold(cleaned, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coordonate_frame = []
        for cnt in contours:
            aria = cv2.contourArea(cnt)
            if 3 < aria < 400:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int (M["m10"] / M["m00"])
                    cY = int (M["m01"] / M["m00"])
                    coordonate_frame.append((cX, cY))

        toate_traiectoriile.append({
            'frame': img_name,
            'celule': coordonate_frame
        })

        print(f"Cadranul {img_name}: am detectat {len(coordonate_frame)} celule.")

    print("Gata procesarea pe toate cadrele! Urmeaza legarea in PyTorch pentru predictie!")
    return toate_traiectoriile

if __name__ == "__main__":
    proceseaza_imagini()



import torch
import torch.nn as nn
import numpy as np

def leaga_traiectorii(coordonate_totale, prag_distanta=30):

    traiectorii_active = []
    urmatorul_id = 0

    for item in coordonate_totale:
        frame_curent = item['celule']
        noi_active = []

        for p_curent in frame_curent:
            cel_mai_apropiat_id = None
            min_dist = float('inf')

            for traj in traiectorii_active:
                ultimul_punct = traj['puncte'][-1]
                dist = np.linalg.norm(np.array(p_curent) - np.array(ultimul_punct))

                if dist < min_dist and dist < prag_distanta:
                    min_dist = dist
                    cel_mai_apropiat_id = traj['id']

            if cel_mai_apropiat_id is not None:
                for traj in traiectorii_active:
                    if traj['id'] == cel_mai_apropiat_id:
                        traj['puncte'].append(p_curent)
                        noi_active.append(traj)
                        break

            else:
                noi_active.append({
                    'id': urmatorul_id,
                    'puncte': [p_curent]
                })
                urmatorul_id += 1

        traiectorii_active = noi_active

    return traiectorii_active



class PredictorTraiectorie():
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super(PredictorTraiectorie, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def antreneaza_modelul_pe_celule(traiectorii):
    model = PredictorTraiectorie()
    criteriu = nn.MSELoss()
    optimizator = torch.optim.Adam(model.parameters(), lr=0.01)

    date_antrenare = [t['puncte'] for t in traiectorii if len(t['puncte']) > 5]

    if not date_antrenare:
        print("Nicio traiectorie suficient de lunga pentru PyTorch inca!")
        return model

    print(f"Am gasit {len(date_antrenare)} traiectorii viabile pentru antrenament PyTorch.")

    return model



import torch
import torch.nn as nn

def leaga_traiectorii(coordonate_totale, prag_distanta=30):
    traiectorii_active = []
    urmatorul_id = 0

    for item in coordonate_totale:
        frame_curent = item['celule']
        noi_active = []

        for p_curent in frame_curent:
            cel_mai_apropiat_id = None
            min_dist = float('inf')

            for traj in traiectorii_active:
                ultimul_punct = traj['puncte'][-1]
                dist = np.linalg.norm(np.array(p_curent) - np.array(ultimul_punct))

                if dist < min_dist and dist < prag_distanta:
                    min_dist = dist
                    cel_mai_apropiat_id = traj['id']

            if cel_mai_apropiat_id is not None:
                for traj in traiectorii_active:
                    if traj['id'] == cel_mai_apropiat_id:
                        traj['puncte'].append(p_curent)
                        noi_active.append(traj)
                        break

            else:
                noi_active.append({
                    'id': urmatorul_id,
                    'puncte': [p_curent]
                })
                urmatorul_id += 1

        traiectorii_active = noi_active

    print(f"Am format {len(traiectorii_active)} traiectorii unice de celule!")
    return traiectorii_active



class PredictorTraiectorie(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2):
        super(PredictorTraiectorie, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    date_brute = proceseaza_imagini()
    traiectorii = leaga_traiectorii(date_brute)
    model = PredictorTraiectorie()
    print("Modelul PyTorch LSTM e pregatit pe datele celulelor!", model)



def antreneaza_pe_bune(model, traiectorii):
    criteriu = nn.MSELoss()
    optimizator = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(10):
        loss_total = 0
        for traj in traiectorii:
            puncte = traj['puncte']
            if len(puncte) < 4:
                continue

            x = torch.tensor(puncte[:-1], dtype=torch.float32).unsqueeze(0)
            y_real = torch.tensor(puncte[-1], dtype=torch.float32).unsqueeze(0)

            optimizator.zero_grad()
            y_pred = model(x)

            loss = criteriu(y_pred, y_real)
            loss.backward()
            optimizator.step()
            loss_total += loss.item()

        print(f"Epoca {epoch+1}, pierdere medie: {loss_total:.4f}")