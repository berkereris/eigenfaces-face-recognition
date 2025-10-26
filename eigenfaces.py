#Berker Eriş 150220315

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error


# Rescaled image uniformly.
def rescale_image(path, size=(112, 92)):
    img = plt.imread(path)
    if img.shape != size:
        h_ratio = size[0] / img.shape[0]
        w_ratio = size[1] / img.shape[1]
        img = zoom(img, (h_ratio, w_ratio))
    return img.flatten()

# Transformed pictures into vectors
def load_ORL(folder="ORL_faces", size=(112, 92)):
    images = []
    labels = []
    for person in range(1, 41):          # 40 kişi
        for img_num in range(1, 11):     # Her kişi için 10 resim
            path = f"{folder}/s{person}/{img_num}.pgm"
            images.append(rescale_image(path, size))
            labels.append(person)
    return np.array(images), np.array(labels)

def saveimage(vec, filename, size=(112, 92)):
    plt.imsave(filename, vec.reshape(size), cmap='gray')

#Loaded and normalized dataset.
data, labels = load_ORL()
mean_face = np.mean(data, axis=0)
normalized = data - mean_face

#Crate sample face with ussing subset of one picture for each person.
sample = [data[i * 10] for i in range(10)]
mean_subset = np.mean(sample, axis=0)

#Created an output file and Saved images
os.makedirs("output", exist_ok=True)
saveimage(mean_face, "output/mean_face_all.png")
saveimage(mean_subset, "output/mean_face_subset.png")









# Compute eigenvalues and eigenfaces thanks to PCA
def find_pca(X, M=None):
    n_samples, n_features = X.shape
    C = np.dot(X, X.T)  # covariance matris
    vals, vecs = np.linalg.eigh(C)  # eigenvalues and eigenvectors
    big_vecs = np.dot(X.T, vecs)
    big_vecs = big_vecs / np.linalg.norm(big_vecs, axis=0)
    idx = np.argsort(vals)[::-1]  # Sorted
    vals = vals[idx]
    big_vecs = big_vecs[:, idx]
    if M:
        vals = vals[:M]
        big_vecs = big_vecs[:, :M]
    return vals, big_vecs


X = normalized #from TASK1
eigenvalues, eigenfaces = find_pca(X)

# Graph eigenvalues
plt.figure()
plt.plot(eigenvalues)
plt.title("Eigenvalues")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.savefig("output/eigenvalues.png")
plt.close()

# Kümülatif varyans grafiği çiz
cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
plt.figure()
plt.plot(cum_var)
plt.title("Cumulative Variance")
plt.xlabel("Number of Eigenfaces (M)")

plt.ylabel("Cumulative Variance")
plt.savefig("output/cumulative_variance.png")
plt.close()

# Visualize top10 eigenfaces.
os.makedirs("output/eigenfaces", exist_ok=True)
for i in range(10):
    saveimage(eigenfaces[:, i], f"output/eigenfaces/ef_top10_{i}.png")



#Created faces for different M values
#NOT: Merhaba Hocam, daha iyi açıklayabilmek için Türkçe yazıyorum. Task2'nin experiment kısmında tam olarak ne istendiğini anlayamadım. M. eigenface görselinin mi, yoksa ilk M eigenface'in ortak oluşturduğu görselin mi istendiği konusunda kararsız kaldım.
#Ben de datasetteki ilk person için ikinci dediğimi uygulamaya karar verdim. Bizde buradan kullandığımız eigenface sayısını arttırdığımızda gerçek resme yaklaştığımızı görmemizi istediğinizi düşündüm.
#Eğer  M. eigenface görselini oluştursaydım M arttıkça görselin temsil kabiliyeti azalacak ve çok "noisy" hale gelecekti. Çünkü, gittikçe gereksiz detaylara daha fazla önem veren, daha önemsiz eigenfaceler(Örneğin 300. eigenface) hesaplamaya dahil edilecekti.
M_values = [10, 20, 50, 100, 200, 300]
for M in M_values:
    _, eigenfaces_M = find_pca(X, M)
    weights = np.dot(X[0] - mean_face, eigenfaces_M)
    combined = np.dot(eigenfaces_M, weights) + mean_face
    saveimage(combined, f"output/eigenfaces/combined_face_M{M}.png")










def reconstruct(face, mean, pcs, M):
    U = pcs[:M]
    diff = face - mean
    w = np.dot(U, diff)
    rec = np.dot(w, U) + mean
    return rec

# Saved original and reconstructed images side by side.
def side_by_side_comparison(orig, rec, filename, shape=(112, 92)):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(orig.reshape(shape), cmap='gray')
    ax[0].set_title("Original")
    ax[0].axis('off')
    ax[1].imshow(rec.reshape(shape), cmap='gray')
    ax[1].set_title("Reconstructed")
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
#Finded the minimum M required to achieve an MSE below 500
def run_reconstruction(data, mean, pcs, labels, shape=(112, 92)):
    os.makedirs("output/reconstructed", exist_ok=True)
    f = open("output/reconstructed/mse_reconstruction.txt", "w")
    Ms = [10, 20, 50, 100, 200, 300]
    people = [10, 11]
    for p in people:
        idx = np.where(labels == p)[0][0]
        face = data[idx]
        mse_below_500 = False
        for M in Ms:
            rec = reconstruct(face, mean, pcs, M)
            mse = mean_squared_error(face, rec)
            name = f"output/reconstructed/comparison_{p}_M{M}.png"
            side_by_side_comparison(face, rec, name, shape)
            f.write(f"s{p}, M={M}, MSE={mse:.2f}\n")
            if not mse_below_500 and mse < 500:
                f.write(f"For s{p}, MSE < 500 was first achieved with M = {M}.\n")
                mse_below_500 = True
    f.close()

run_reconstruction(data, mean_face, eigenfaces.T, labels)











def projection(data, mean, pcs, M):
    U = pcs[:M]
    return np.dot(data - mean, U.T)

# Classificted
def classify(proj, labels):
    classes = np.unique(labels)
    centers = []
    for c in classes:
        class_proj = proj[labels == c]
        center = np.mean(class_proj, axis=0)
        centers.append(center)
    centers = np.array(centers)
    preds = []
    for p in proj:
        dists = np.linalg.norm(centers - p, axis=1)
        pred = np.argmin(dists) + 1
        preds.append(pred)
    return np.array(preds)

# Saved confusion matrix.
def confision_matrix(true, pred, M, folder="output/recognition"):
    os.makedirs(folder, exist_ok=True)
    cm = confusion_matrix(true, pred, labels=np.arange(1, 41))
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='gray')
    plt.title(f"Confusion Matrix (M={M})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()
    plt.savefig(f"{folder}/confusion_matrix_M{M}.png")
    plt.close()

# Finded te best accuracy for different Ms.
def run_experiment(data, mean, pcs, labels):
    Ms = [10, 20, 50, 100, 200, 300]
    accs = []
    for M in Ms:
        proj = projection(data, mean, pcs, M)
        preds = classify(proj, labels)
        acc = accuracy_score(labels, preds)
        accs.append(acc)
        confision_matrix(labels, preds, M)
    best_idx = np.argmax(accs)
    best_M = Ms[best_idx]
    best_acc = accs[best_idx] * 100
    with open("output/recognition/best_accuracy.txt", "w") as f:
        f.write(f"Best accuracy is %{best_acc:.2f} with M = {best_M}\n")
    return Ms, accs


def plot_accuracy(Ms, accs):
    plt.plot(Ms, [a * 100 for a in accs], marker='o')
    plt.xlabel("Number of Eigenface (M)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs M")
    plt.grid(True)
    plt.savefig("output/recognition/accuracy_vs_eigenfaces.png")
    plt.close()

Ms, accs = run_experiment(data, mean_face, eigenfaces.T, labels)
plot_accuracy(Ms, accs)










def run_noise_test(data, mean, pcs, labels, noise='gaussian', M=50, shape=(112, 92)):
    os.makedirs("output/noise", exist_ok=True)
    levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    accs = []
    np.random.seed(42)
    chosen = np.random.choice(np.unique(labels), 10, replace=False)
    test_idx = []
    for p in chosen:
        idxs = np.where(labels == p)[0]
        test_idx += list(idxs[:3])
    train_idx = [i for i in range(len(labels)) if i not in test_idx]
    test_x = data[test_idx]
    test_y = labels[test_idx]
    train_x = data[train_idx]
    train_y = labels[train_idx]
    proj_train = projection(train_x, mean, pcs, M)
    for i, lvl in enumerate(levels):
        noisy = []
        for j, img in enumerate(test_x):
            img2d = img.reshape(shape)
            if noise == 'gaussian':
                n = np.random.normal(0, lvl * 255, img2d.shape)
                img_noisy = img2d + n
                img_noisy = np.clip(img_noisy, 0, 255)
            else:
                img_noisy = np.copy(img2d).flatten()
                s = int(lvl * img_noisy.size * 0.5)
                p = int(lvl * img_noisy.size * 0.5)
                salt = np.random.randint(0, img_noisy.size, s)
                pepper = np.random.randint(0, img_noisy.size, p)
                img_noisy[salt] = 255
                img_noisy[pepper] = 0
                img_noisy = img_noisy.reshape(shape)
            noisy.append(img_noisy.flatten())
            name = f"output/noise/noisy_{noise}_{str(lvl).replace('.', '')}_{j+1}.png"
            plt.imsave(name, img_noisy, cmap='gray')
        noisy = np.array(noisy)
        proj_noisy = projection(noisy, mean, pcs, M)
        preds = classify_noise(proj_noisy, proj_train, train_y)
        acc = accuracy_score(test_y, preds)
        accs.append(acc)
    return levels, accs

# Accuracy vs noise graph
def plot_noise_acc(levels, accs, noise):
    plt.plot(levels, [a * 100 for a in accs], marker='o')
    plt.xlabel("Noise")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{noise.capitalize()} Accuracy vs Noise")
    plt.grid(True)
    plt.savefig(f"output/noise/accuracy_vs_noise_{noise}.png")
    plt.close()

def classify_noise(test_proj, train_proj, train_labels):
    classes = np.unique(train_labels)
    centers = []
    for c in classes:
        class_proj = train_proj[train_labels == c]
        center = np.mean(class_proj, axis=0)
        centers.append(center)
    centers = np.array(centers)
    preds = []
    for p in test_proj:
        dists = np.linalg.norm(centers - p, axis=1)
        pred = np.argmin(dists) + 1
        preds.append(pred)
    return np.array(preds)

# Compare different noises.
levels, acc_gauss = run_noise_test(data, mean_face, eigenfaces.T, labels, noise='gaussian')
plot_noise_acc(levels, acc_gauss, 'gaussian')

levels, acc_sp = run_noise_test(data, mean_face, eigenfaces.T, labels, noise='saltpepper')
plot_noise_acc(levels, acc_sp, 'saltpepper')