# Task 6: K-Nearest Neighbors (KNN) Classification

## 🎯 Objective  
Implement and understand KNN for a multi-class classification problem (Iris dataset).

---

## 🛠️ Tools & Libraries Used  
- Python 3  
- Pandas   
- NumPy   
- Matplotlib   
- Seaborn   
- Scikit-learn (KNeighborsClassifier)  

---

## 📂 Dataset  
**Iris Detection Dataset**  
- 150 samples  
- Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm  
- Target: Species (Iris-setosa, Iris-versicolor, Iris-virginica)  

---

## 🧪 Steps Performed

1. **Load & Inspect**  
   - Dropped the `Id` column.  
   - Checked class distribution (50 samples each of three species).  

2. **Label Encoding**  
   - Converted categorical `Species` to numeric:  
     - Iris-setosa → 0  
     - Iris-versicolor → 1  
     - Iris-virginica → 2  

3. **Train/Test Split**  
   - Split data into train (80%) and test (20%), stratifying by `Species` to keep class proportions.  

4. **Feature Scaling**  
   - Standardized features to mean = 0, std = 1 using `StandardScaler` (important for distance calculations in KNN).  

5. **KNN Training & Tuning**  
   - For k in 1…15, trained `KNeighborsClassifier` on the training set and recorded test accuracy.  
   - Found that **k = 5** gave the highest accuracy on the test set.  

6. **Evaluation (k = 5)**  
   - Test accuracy: _0.967_  
   - Confusion matrix:  
     - Iris-setosa → always correct  
     - Some confusion between Iris-versicolor & Iris-virginica  
   - Classification report (precision, recall, F1) for each class.  

7. **Decision Boundary Visualization**  
   - Used only `PetalLengthCm` & `PetalWidthCm` (2D) for plotting.  
   - Created a meshgrid, predicted each point’s class, and plotted colored regions showing how KNN partitions space.  
   - Overlayed actual training points to confirm model’s decision regions.  

---

## 📈 Results

- **Optimal k**: 5  
- **Test Accuracy**: 0.967  
- **Precision / Recall / F1** for each species (see notebook).  
- **Decision Boundary**: Clear separation in 2D (Petal features).

---

## 🔍 What I Learned

- **KNN Algorithm**: How it classifies a point by looking at the “k” nearest neighbors in feature space (Euclidean distance).  
- **Choosing k**: Too small → high variance (overfitting). Too large → high bias (underfitting). We find a sweet spot by testing multiple k.  
- **Normalization**: Crucial because KNN relies on distance metrics—features with larger numeric ranges will otherwise dominate.  
- **Decision Boundaries**: How the algorithm partitions the feature space into regions for each class.  
- **Multi-class Handling**: KNN naturally extends to multiple classes by simple majority vote among neighbors.  
