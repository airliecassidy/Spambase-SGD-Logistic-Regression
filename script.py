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
              print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

      return self


  def predict(self, x):
    z = np.dot(x, self.w) + self.b
    probabilities = logistic(z)
    return (probabilities >= 0.5).astype(int)

  def predict_prob(self, x):
    z = np.dot(x, self.w) + self.b
    return logistic(z)

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
X_cv = X_train[inds]
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

eta = 0.1 #learning rate
B = 16
epochs = 200 # epoches

lambda_grid = np.array([0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
print(f"lambda_grid: {lambda_grid}")


#def train_and_evaluate_model(X_train, y_train, X_val, y_val, lambda_val, learning_rate, batch_size, epochs):


def k_fold_cv(X, y, k=5):
  indices = np.arange(len(X))
  np.random.shuffle(indices)
  fold_sizes = np.full(k, len(X) % k)
  fold_sizes[:len(X)] += 1
  current = 0
  folds = []

  for fold_size in fold_sizes: # this is splitting indices into k folds
    start, stop = current, current + fold_size
    folds.append(indices[start:stop])
    current = stop

  scores = []
  for i in range(k):
    test_indices = folds[i]
    train_indices = np.hstack([folds[j] for j in range(k) if j != i])

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
    print(f"Fold {i+1}: Train size={len(X_train)}, Test size={len(X_test)}")
  return "CV Complete"

