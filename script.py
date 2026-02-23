import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 2026
rng = np.random.default_rng(RANDOM_SEED)

# hyperparameters to try initially
BATCH_SIZES = [1, 16, 64]
LEARNING_RATE = 0.01
EPOCHS = 100
LAMBDA = 0.001

# loading the data
data_path = '/Users/airliecassidy/PyCharmMiscProject/spambase.data'
column_names = [f'feature_{i}' for i in range(57)] + ['is_spam']
df = pd.read_csv(data_path, header=None, names=column_names)

# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# checking the dimensions
print(df.shape)
print(f"\nFeatures: {X.shape}")
print(f"Target: {y.shape}")

# standardizing function
def standardize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    # divide by 0 case
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, mean, std


# Train and test split (5% train, 95% test)
def train_test_split(X, y, train_size=0.05, random_state=2026):
    rng_split = np.random.default_rng(random_state)
    n_samples = len(y)

    # stratified split to maintain balance
    indices_class_0 = np.where(y == 0)[0]
    indices_class_1 = np.where(y == 1)[0]

    rng_split.shuffle(indices_class_0)
    rng_split.shuffle(indices_class_1)

    # split
    n_train_0 = int(len(indices_class_0) * train_size)
    n_train_1 = int(len(indices_class_1) * train_size)

    train_indices = np.concatenate([indices_class_0[:n_train_0], indices_class_1[:n_train_1]])
    test_indices = np.concatenate([indices_class_0[n_train_0:], indices_class_1[n_train_1:]])

    # shuffle train and test indices
    rng_split.shuffle(train_indices)
    rng_split.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# apply train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.05, random_state=RANDOM_SEED)
# standardize
X_train_scaled, X_test_scaled, train_mean, train_std = standardize(X_train, X_test)

# check for before and after standardization
print("Preprocessing:")
print(f"Training Samples: {X_train.shape[0]}, Spam:({y_train.mean():.2%})")
print(f"Test Samples: {X_test.shape[0]}, Spam: ({y_test.mean():.2%})")

print(f"\nStandardization:")
print(f"  Training mean: {X_train_scaled.mean():.6f}")
print(f"  Training std: {X_train_scaled.std():.6f}")

"""### Task 1: Logistic Regression with SGD"""

def logistic(z):
  return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class LogisticRegressionSGD:
  def __init__(self, learning_rate=0.01, batch_size=16, epochs=100, lambda_reg=0.0, random_state=2026):
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.epochs = epochs
    self.lambda_reg = lambda_reg
    self.random_state = random_state
    self.rng = np.random.default_rng(random_state)

    self.w = None
    self.b = None
    self.train_loss_history = []
    self.train_acc_history = []

  def cost(self, x, y):
    N = x.shape[0]
    z = np.dot(x, self.w) + self.b

    cross_entropy = np.mean(y * np.log1p(np.exp(-z)) + (1 - y) * np.log1p(np.exp(z)))

    # L2 regularization penalty (with bias excluded)
    l2_penalty = self.lambda_reg * np.sum(self.w ** 2)
    return cross_entropy + l2_penalty

  def gradient(self, x, y):
    m = x.shape[0]
    z = np.dot(x, self.w) + self.b
    yh = logistic(z)

    # gradient with respect to weights
    grad_w = np.dot(x.T, yh - y) / m + 2 * self.lambda_reg * self.w
    # gradient with respect to bias (no L2 on bias)
    grad_b = np.sum(yh - y) / m

    return grad_w, grad_b

  def fit(self, x, y, verbose=True):
    N, D = x.shape
    # initialize weights to 0
    self.w = np.zeros(D)
    self.b = 0.0

    self.train_loss_history = []
    self.train_acc_history = []

    # Mini-batch SGD loop
    for epoch in range(self.epochs):
      indices = self.rng.permutation(N)
      x_shuffled = x[indices]
      y_shuffled = y[indices]

      # process the mini-batches
      for i in range(0, N, self.batch_size):
        end_index = min(i + self.batch_size, N)
        x_batch = x_shuffled[i:end_index]
        y_batch = y_shuffled[i:end_index]

        # compute the gradients
        grad_w, grad_b = self.gradient(x_batch, y_batch)

        # update the weights
        self.w = self.w - self.learning_rate * grad_w
        self.b = self.b - self.learning_rate * grad_b

      # tracking the metrics
      train_loss = self.cost(x, y)
      train_acc = np.mean(self.predict(x) == y)
      self.train_loss_history.append(train_loss)
      self.train_acc_history.append(train_acc)

      if verbose and (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

    return self

  def predict(self, x):
    z = np.dot(x, self.w) + self.b
    probabilities = logistic(z)
    return (probabilities >= 0.5).astype(int)

  def predict_prob(self, x):
    z = np.dot(x, self.w) + self.b
    return logistic(z)

"""## Experiments"""

# First try: No regularization
print("EXPERIMENT 1: NO REGULARIZATION (λ = 0)")
model_no_reg = LogisticRegressionSGD(
    learning_rate=LEARNING_RATE,
    batch_size=16,
    epochs=EPOCHS,
    lambda_reg=0.0,
    random_state=RANDOM_SEED
)

model_no_reg.fit(X_train_scaled, y_train, verbose=True)
train_acc_no_reg = np.mean(model_no_reg.predict(X_train_scaled) == y_train)
test_acc_no_reg = np.mean(model_no_reg.predict(X_test_scaled) == y_test)

print(f"RESULTS:")
print(f"Train Acc: {train_acc_no_reg:.4f}")
print(f"Test Acc: {test_acc_no_reg:.4f}")

# Second try: with regularization
print("EXPERIMENT 2: L2 REGULARIZATION (λ = 0.001)")
model_l2 = LogisticRegressionSGD(
    learning_rate=LEARNING_RATE,
    batch_size=16,
    epochs=EPOCHS,
    lambda_reg=LAMBDA,
    random_state=RANDOM_SEED
)

model_l2.fit(X_train_scaled, y_train, verbose=True)

train_acc_l2 = np.mean(model_l2.predict(X_train_scaled) == y_train)
test_acc_l2 = np.mean(model_l2.predict(X_test_scaled) == y_test)

print(f"RESULTS:")
print(f"Train Acc: {train_acc_l2:.4f}")
print(f"Test Acc: {test_acc_l2:.4f}")

# Third: comparing the batch sizes
print("EXPERIMENT 3: BATCH SIZE COMPARISON")
batch_results = {}

for batch in BATCH_SIZES:
  print(f"\nBatch size {batch}...")

  model = LogisticRegressionSGD(
      learning_rate=LEARNING_RATE,
      batch_size=batch,
      epochs=EPOCHS,
      lambda_reg=0.0,
      random_state=RANDOM_SEED
  )

  model.fit(X_train_scaled, y_train, verbose=False)
  train_acc = np.mean(model.predict(X_train_scaled) == y_train)
  test_acc = np.mean(model.predict(X_test_scaled) == y_test)

  print(f"Train: {train_acc:.4f}, Test: {test_acc:.4f}")
  batch_results[batch] = {'model': model, 'train_acc': train_acc, 'test_acc': test_acc}

"""## Training Curves"""

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

epochs_range = range(1, EPOCHS + 1)

# Loss: No Regularization vs L2
ax1 = axes[0, 0]
ax1.plot(epochs_range, model_no_reg.train_loss_history,
         label='No Regularization (λ=0)', linewidth=2, color='#e74c3c')
ax1.plot(epochs_range, model_l2.train_loss_history,
         label='L2 (λ=0.001)', linewidth=2, color='#3498db')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss: No Regularization vs L2', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Accuracy: No Regularization vs L2
ax2 = axes[0, 1]
ax2.plot(epochs_range, model_no_reg.train_acc_history,
         label='No Regularization (λ=0)', linewidth=2, color='#e74c3c')
ax2.plot(epochs_range, model_l2.train_acc_history,
         label='L2 (λ=0.001)', linewidth=2, color='#3498db')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Training Accuracy', fontsize=12)
ax2.set_title('Training Accuracy: No Regularization vs L2', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Batch comparison: Loss
ax3 = axes[1, 0]
colors = ['#2ecc71', '#f39c12', '#9b59b6']
for (bs, data), color in zip(batch_results.items(), colors):
    ax3.plot(epochs_range, data['model'].train_loss_history,
            label=f'Batch={bs}', linewidth=2, color=color)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Training Loss', fontsize=12)
ax3.set_title('Batch Size: Loss', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Batch comparison: Accuracy
ax4 = axes[1, 1]
for (bs, data), color in zip(batch_results.items(), colors):
    ax4.plot(epochs_range, data['model'].train_acc_history,
            label=f'Batch={bs}', linewidth=2, color=color)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Training Accuracy', fontsize=12)
ax4.set_title('Batch Size: Accuracy', fontsize=14, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""###Weight Distribution Plot: No regularization vs L2"""

plt.figure(figsize=(10, 6))
plt.hist(model_no_reg.w, bins=30, alpha=0.6, color='#e74c3c',
         label=f'No Regularization (mean |w|={np.mean(np.abs(model_no_reg.w)):.3f})',
         edgecolor='black')
plt.hist(model_l2.w, bins=30, alpha=0.6, color='#3498db',
         label=f'L2 (mean |w|={np.mean(np.abs(model_l2.w)):.3f})',
         edgecolor='black')
plt.xlabel('Weight Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Effect of L2 Regularization on Learned Weights', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""###Task 2: Hyperparameter Tuning with K-Fold Cross-Validation"""

# Hyperparameters
# Hyperparameters
learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001]
batches = [1, 16, 64]
epochs = [50, 100, 200]
num_folds = 5
LAMBDA_CV = 0.001

# cross-validate
def cross_validate(n, n_folds=5):
  n_val = n // n_folds
  for f in range(n_folds):
    val_inds = list(range(f * n_val, (f + 1) * n_val))
    tr_inds = list(range(f * n_val)) + list(range((f + 1) * n_val, n))
    yield tr_inds, val_inds

# shuffle training data before CV
n_train = len(y_train)
rng_cv = np.random.default_rng(RANDOM_SEED)
inds = rng_cv.permutation(n_train)
X_cv = X_train_scaled[inds] # CHANGED TO X_train_scaled
y_cv = y_train[inds]

configs = [(lr, bs, ep)
           for lr in learning_rates
           for bs in batches
           for ep in epochs]

n_configs = len(configs)
err_loss = np.zeros((n_configs, num_folds))
err_acc = np.zeros((n_configs, num_folds))

for i, (lr, bs, ep) in enumerate(configs):
  for f, (tr, val) in enumerate(cross_validate(n_train, num_folds)):
    # standardize on the fold's training split
    mean_f = np.mean(X_cv[tr], axis=0)
    std_f = np.std(X_cv[tr],  axis=0)
    std_f[std_f == 0] = 1.0
    X_tr_s = (X_cv[tr]  - mean_f) / std_f
    X_val_s = (X_cv[val] - mean_f) / std_f

    # training and evaluate
    model = LogisticRegressionSGD(learning_rate=lr, batch_size=bs, epochs=ep, lambda_reg=LAMBDA_CV, random_state=RANDOM_SEED)
    model.fit(X_tr_s, y_cv[tr], verbose=False)

    err_loss[i, f] = model.cost(X_val_s, y_cv[val])
    err_acc[i,  f] = np.mean(model.predict(X_val_s) == y_cv[val])

print(f"\n CV: ({n_configs} configs × {num_folds} folds)")

"""### Results Table"""

mean_loss = np.mean(err_loss, axis=1)
std_loss = np.std(err_loss,  axis=1)
mean_acc = np.mean(err_acc,  axis=1)
std_acc = np.std(err_acc,   axis=1)

order = np.argsort(mean_loss)

print(f"{'η':>8}  {'B':>4}  {'Ep':>4}  {'Mean Loss':>9}  {'Std Loss':>9}  {'Mean Acc':>9}  {'Std Acc':>8}")
for i in order:
  lr, bs, ep = configs[i]
  print(f"{lr:>8.4f}  {bs:>4}  {ep:>4}"
  f"{mean_loss[i]:>9.4f}  {std_loss[i]:>9.4f}"
  f"{mean_acc[i]:>9.4f}  {std_acc[i]:>8.4f}")

# best config: only consider the ones with SD lower than 0.15
stable = np.where(std_loss < 0.15)[0]
best_i = stable[np.argmin(mean_loss[stable])]

BEST_LEARNING_RATE, BEST_BATCH_SIZE, BEST_EPOCHS = configs[best_i]

print(f"\nBest config: η={BEST_LEARNING_RATE}, B={BEST_BATCH_SIZE}, epochs={BEST_EPOCHS}")
print(f"CV loss: {mean_loss[best_i]:.4f} ± {std_loss[best_i]:.4f}")
print(f"CV acc:  {mean_acc[best_i]:.4f} ± {std_acc[best_i]:.4f}")

"""## Plot of Mean CV Loss vs Learning Rate"""

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#e74c3c', '#3498db', '#2ecc71']

for ax, ep in zip(axes, [50, 200]):
  for bs, color in zip(batches, colors):
    # Indices for this (ep, bs) slice
    idx = [j for j, (lr, b, e) in enumerate(configs) if b == bs and e == ep]
    ax.errorbar(range(len(learning_rates)),mean_loss[idx],yerr=std_loss[idx], label=f'B={bs}', marker='o', linewidth=2, capsize=4, color=color)
    ax.set_xticks(range(len(learning_rates)))
    ax.set_xticklabels([str(lr) for lr in learning_rates], rotation=20)
    ax.set_xlabel('Learning rate (η)')
    ax.set_ylabel('Mean CV loss')
    ax.set_title(f'CV Loss vs η  (epochs={ep})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

"""###Heatmap of hyperparameters"""

# to show how learning rate and batch size interact
fig, ax = plt.subplots(figsize=(8, 6))

# Build matrix: rows = learning rates, cols = batch sizes
heatmap_data = np.zeros((len(learning_rates), len(batches)))
for i, lr in enumerate(learning_rates):
    for j, bs in enumerate(batches):
        # Average loss across all epoch settings for this (lr, bs) pair
        idx = [k for k, (l, b, e) in enumerate(configs) if l == lr and b == bs]
        heatmap_data[i, j] = np.mean(mean_loss[idx])

im = ax.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(batches)))
ax.set_yticks(range(len(learning_rates)))
ax.set_xticklabels(batches)
ax.set_yticklabels(learning_rates)
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('CV Loss Heatmap: Learning Rate × Batch Size\n(averaged over epochs)',
             fontsize=13, fontweight='bold')

# Add values in cells
for i in range(len(learning_rates)):
    for j in range(len(batches)):
        ax.text(j, i, f'{heatmap_data[i, j]:.3f}',
                ha='center', va='center', fontsize=10,
                color='white' if heatmap_data[i, j] > 0.6 else 'black')

plt.colorbar(im, ax=ax, label='Mean CV Loss')
plt.tight_layout()
plt.show()

"""# Task 3: Bias-Variance Trade off via λ Sweep"""

import matplotlib.pyplot as plt

ETA = 0.1 #learning rate
BATCH_SIZE = 16
EPOCHS = 200 # epoches
K_FOLDS = 10

lambda_grid = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
print(f"lambda_grid: {lambda_grid}")


cv_results = {
    'lambda': [],
    'train_loss_mean': [],
    'val_loss_mean': [],
    'train_acc_mean': [],
    'val_acc_mean': []
    }

n_train = len(y_train)
rng_cv = np.random.default_rng(RANDOM_SEED)
indices = rng_cv.permutation(n_train)
fold_size = n_train // K_FOLDS

#lambda sweep with k-fold cv
for lambda_val in lambda_grid:
  fold_train_losses = []
  fold_val_losses = []
  fold_train_accs = []
  fold_val_accs = []

  for fold in range(K_FOLDS):
    start = fold * fold_size
    end = start + fold_size if fold < K_FOLDS - 1 else n_train
    val_indices = indices[start:end]
    train_indices = np.concatenate([indices[:start], indices[end:]])

#standardization process
    mean = np.mean(X_train[train_indices], axis=0)
    std  = np.std(X_train[train_indices], axis=0)
    std[std == 0] = 1.0
    X_tr  = (X_train[train_indices] - mean) / std
    X_val = (X_train[val_indices]   - mean) / std

    model = LogisticRegressionSGD (
        learning_rate = ETA,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        lambda_reg = lambda_val,
        random_state=RANDOM_SEED
    )

    model.fit(X_tr, y_train[train_indices], verbose=False)

#evaluation stage
    fold_train_losses.append(model.cost(X_tr, y_train[train_indices]))
    fold_val_losses.append(model.cost(X_val, y_train[val_indices]))
    fold_train_accs.append(np.mean(model.predict(X_tr) == y_train[train_indices]))
    fold_val_accs.append(np.mean(model.predict(X_val) == y_train[val_indices]))

#recording the mean across folds
  cv_results['lambda'].append(lambda_val)
  cv_results['train_loss_mean'].append(np.mean(fold_train_losses))
  cv_results['val_loss_mean'].append(np.mean(fold_val_losses))
  cv_results['train_acc_mean'].append(np.mean(fold_train_accs))
  cv_results['val_acc_mean'].append(np.mean(fold_val_accs))
  print(f"λ={lambda_val:.0e}  |  "
        f"Train Loss: {cv_results['train_loss_mean'][-1]:.4f}  |  "
        f"Val Loss: {cv_results['val_loss_mean'][-1]:.4f}  |  "
        f"Val Acc: {cv_results['val_acc_mean'][-1]:.4f}")

#plot results
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

lambda_arr = np.array(cv_results['lambda'])
lambda_plot = np.where(lambda_arr == 0, 1e-8, lambda_arr)   # display only

axes[0].semilogx(cv_results['lambda'], cv_results['train_loss_mean'], 'o-', label='Train', linewidth=2)

axes[0].semilogx(cv_results['lambda'], cv_results['val_loss_mean'], 's-', label='Val',linewidth=2)
axes[0].set_xlabel('λ (log scale)', fontsize=12); axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12); axes[0].legend(); axes[0].grid(
    True, alpha=0.3)
axes[0].set_title('Loss vs λ')
# Annotate the λ=0 tick clearly
axes[0].axvline(x=1e-8, color='grey', linestyle='--', alpha=0.4, label='λ=0 (no reg)')

# Accuracy plot
axes[1].semilogx(cv_results['lambda'], cv_results['train_acc_mean'], 'o-', label='Train', linewidth=2)
axes[1].semilogx(cv_results['lambda'], cv_results['val_acc_mean'],'s-', label='Val',   linewidth=2)
axes[1].set_xlabel('λ (log scale)', fontsize=12); axes[1].set_ylabel('Accuracy', fontsize=12); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title('Accuracy vs λ', fontsize=13, fontweight='bold')

plt.tight_layout(); plt.show()

best_idx    = np.nanargmin(cv_results['val_loss_mean'])
BEST_LAMBDA = cv_results['lambda'][best_idx]

print(f"\nBest λ (from CV): {BEST_LAMBDA:.2e}")
print(f"  CV Val Loss: {cv_results['val_loss_mean'][best_idx]:.4f}")
print(f"  CV Val Acc:  {cv_results['val_acc_mean'][best_idx]:.4f}")

# Retrain on all training data with best lambda
# Standardize once from the full raw training set
final_mean = np.mean(X_train, axis=0)
final_std  = np.std( X_train, axis=0)
final_std[final_std == 0] = 1.0
X_train_final = (X_train - final_mean) / final_std
X_test_final  = (X_test  - final_mean) / final_std   # use same stats for test

final_model = LogisticRegressionSGD(
    learning_rate = ETA,
    batch_size    = BATCH_SIZE,
    epochs        = EPOCHS,
    lambda_reg    = BEST_LAMBDA,
    random_state  = RANDOM_SEED,
)
final_model.fit(X_train_final, y_train, verbose=True)

final_test_loss = final_model.cost(X_test_final, y_test)
final_test_acc  = np.mean(final_model.predict(X_test_final) == y_test)

print(f"\n── Final Test Results (λ = {BEST_LAMBDA:.2e}) ──")
print(f"  Test Cross-Entropy Loss : {final_test_loss:.4f}")
print(f"  Test Accuracy           : {final_test_acc:.4f}")

"""# Task 4: Task 4: L1-Regularized Logistic Regression and the Regularization Path (Using scikit-learn)"""

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

data_path = '/content/drive/MyDrive/comp551a2_data/spambase.data'
Cs = np.logspace(-1, 1, 30)
scaler = StandardScaler()
X_train_scaled_2 = X_train
result_cols = ['penalty','coef_variance', 'num_features']
results = pd.DataFrame(columns=result_cols)

for C in Cs[:5]:
    # single logistic regression model
    logisticRegr = LogisticRegression(penalty="l1",
                                      solver="saga",
                                      C=C,
                                      max_iter=5000,
                                      warm_start=True)
    logisticRegr.fit(X_train_scaled_2, y_train)
    coefficients = logisticRegr.coef_
    variance_coef = np.var(coefficients)
    num_features = np.sum(coefficients != 0)
    result = [C, variance_coef, num_features]
    results.loc[len(results)] = result


# logistic regression with k-fold cross validation
logisticRegrKfold = LogisticRegressionCV(penalty="l1",
                                    solver="saga",
                                    Cs=Cs,
                                    max_iter=5000,
                                    )
logisticRegrKfold.fit(X_train_scaled_2, y_train)
logisticRegrKfold.scores_

coef_path = []

for C in Cs:
    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=C,
        max_iter=5000
    )
    model.fit(X_train_scaled_2, y_train)
    coef_path.append(model.coef_.ravel())

coef_path = np.array(coef_path)

# Select top-k coefficients by max magnitude
k = 10
max_abs_coef = np.max(np.abs(coef_path), axis=0)
top_k_indices = np.argsort(max_abs_coef)[-k:]

plt.figure()
for idx in top_k_indices:
    plt.plot(Cs, coef_path[:, idx])

plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Coefficient Value")
plt.title("Regularization Path (Top-10 Coefficients)")
plt.show()

nonzero_counts = np.sum(coef_path != 0, axis=1)

plt.figure()
plt.plot(Cs, nonzero_counts)
plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Number of Non-Zero Coefficients")
plt.title("Sparsity vs Regularization Strength")
plt.show()

logistic_cv = LogisticRegressionCV(
    penalty="l1",
    solver="saga",
    Cs=Cs,
    cv=5,
    max_iter=5000
)

logistic_cv.fit(X_train_scaled_2, y_train)

# Extract mean CV accuracy across folds
scores = logistic_cv.scores_[list(logistic_cv.scores_.keys())[0]]
mean_cv_scores = np.mean(scores, axis=0)

plt.figure()
plt.plot(Cs, mean_cv_scores)
plt.xscale("log")
plt.xlabel("C (log scale)")
plt.ylabel("Mean CV Accuracy")
plt.title("Cross-Validation Accuracy vs C")
plt.show()

print("Best C from CV:", logistic_cv.C_[0])
print("Best CV accuracy:", np.max(mean_cv_scores))
