#multiclass classification problem where 3 classes are: Type_1 Type_2 Type_3

#SPLIT, TRAIN & VALIDATION

## random split
train_df, valid_df = train_test_split(df, test_size=0.3, random_state= 42)
train_df.reset_index(inplace = True)
valid_df.reset_index(inplace = True)

# create dataset for validation & train
train_dataset = CancerDataset(train_df, augmentations = transform) 
valid_dataset = CancerDataset(valid_df)

# create dataloaders
train_dataloader = DataLoader(train_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = False)

valid_dataloader = DataLoader(valid_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = False)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        bce = self.cross_entropy_loss(inputs, targets)
        pt = torch.exp(-bce)
        loss = bce * self.alpha * (torch.pow((1 - pt), self.gamma))
        return loss

# in general alpha should be decreasing and gamma should be increasing.
class Model(nn.Module):
    def __init__(self, model_name, pretrained = True, num_classes = 3):
        super().__init__()
        self.model_name = model_name
        self.cnn = timm.create_model(self.model_name, pretrained = pretrained, num_classes = num_classes)

    def forward(self, x):
        x = self.cnn(x)
        return x
    
    def train_mode(self):
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_acc = 0
        self.train_loss_history = []
        self.train_acc_history = []
        
    def valid_mode(self):
        self.valid_loss_history = []
        self.valid_acc_history = []
        
def train_one_epoch(train_loader, model, criterion, optimizer, device):
    # switch to train mode
    model.train()   
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    loss, correct = 0, 0
    ################################# train #################################
    for batch, (x, y) in enumerate(train_loader):
        start = time.time()
        device = torch.device(device)
        x, y = x.to(device), y.to(device)  
        # compute predictions and loss
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y.long().squeeze()) 
        current = batch * len(x)
        # Backpropagation: only in train function, not done in validation function
        loss.backward()
        optimizer.step()
        # sum correct predictions
        y_pred, y_true = torch.argmax(pred, axis=1), y.long().squeeze()
        correct += (y_pred == y_true).type(torch.float).sum().item()
        end = time.time()
        time_delta = np.round(end - start, 3)
        # log
        loss, current = np.round(loss.item(), 5), batch * len(x)
    # metrics: calculate accuracy and loss for epoch (all batches)
    correct /= size # epoch accuracy
    loss /= num_batches # epoch loss
    print(f"Train: Accuracy: {(100*correct):>0.2f}%, Avg loss: {loss:>5f} \n")
    model.train_loss_history.append(loss)
    model.train_acc_history.append(correct)
    return loss, correct
    
def valid_one_epoch(valid_loader, model, criterion, device):
    model.eval()
    size = len(valid_loader.dataset)
    num_batches = len(valid_loader)
    loss, correct = 0, 0
    ################################# validation #################################
    with torch.no_grad(): # disable gradients
        for batch, (x, y) in enumerate(valid_loader):
            start = time.time()
            device = torch.device(device)
            x, y = x.to(device), y.to(device)
            # compute predictions and loss
            pred = model(x)
            loss = criterion(pred, y.long().squeeze()) 
            current = batch * len(x)
            # sum correct predictions
            y_pred, y_true = torch.argmax(pred, axis=1), y.long().squeeze()
            correct += (y_pred == y_true).type(torch.float).sum().item()
            end = time.time()
            time_delta = np.round(end - start, 3)
            # log
            loss, current = np.round(loss.item(), 5), batch * len(x)
    # metrics: calculate accuracy and loss for epoch (all batches)
    correct /= size # epoch accuracy
    loss /= num_batches # epoch loss
    model.valid_loss_history.append(loss)
    model.valid_acc_history.append(correct)
    print(f"Valid: Accuracy: {(100*correct):>0.2f}%, Avg loss: {loss:>5f} \n")
    return loss, correct

def train_valid(train_loader,valid_loader, model, device):
    # Create optimizer & loss
    model.optimizer = Adam(model.parameters(),lr=1e-4)
    loss_fn = FocalLoss()
    
    print('\n ******************************* Using backbone: ', model.model_name, " ******************************* \n")
    print('Starting Training...\n')
    start_train_time = time.time()
    model.train_mode()
    model.valid_mode()
    for epoch in tqdm(range(0, N_EPOCHS)):
        print(f"\n-------------------------------   Epoch {epoch + 1}   -------------------------------\n")
        start_epoch_time = time.time()
        # train
        train_one_epoch(train_loader, model, loss_fn, model.optimizer, device)
        # validation
        valid_loss, valid_acc = valid_one_epoch(valid_loader, model, loss_fn, device)
        # save validation loss if it was improved (reduced) & validation accuracy if it was improved (increased)
        if valid_loss < model.best_loss and valid_acc > model.best_acc:
            model.best_epoch = epoch + 1
            model.best_loss = valid_loss
            model.best_acc = valid_acc
            # save the model's weights and biases   
            torch.save(model.state_dict(), OUTPUT_PATH + f"{model.model_name}_ep{model.best_epoch}.pth")        
            torch.save(model.state_dict(), OUTPUT_PATH + f"{model.model_name}_ep{model.best_epoch}.pth")

        end_epoch_time = time.time()
        time_delta = np.round(end_epoch_time - start_epoch_time, 3)
        print("\n\nEpoch Elapsed Time: {} s".format(time_delta))

    end_train_time = time.time()
    print("\n\nTotal Elapsed Time: {} min".format(np.round((end_train_time - start_train_time)/60, 3)))
    print("Done!")

def plot_results(model):
    fig = plt.figure(figsize = (18, 8))
    fig.suptitle(f"{model.model_name} Training Results", fontsize = 18)

    space = np.arange(1, N_EPOCHS + 1, 1)
    if N_EPOCHS <= 20:
        x_ticks = np.arange(1, N_EPOCHS + 1, 1)
    else:
        x_ticks = np.arange(1, N_EPOCHS + 1, int(N_EPOCHS/20) + 1)

    # Loss plot
    ax1 = plt.subplot(1, 2, 1) 
    ax1.plot(space, model.train_loss_history, label='Training', color = 'black')
    ax1.plot(space, model.valid_loss_history, label='Validation', color = 'blue')
    plt.xticks(x_ticks)
    plt.axhline(0, linestyle = 'dashed', color = 'grey')
    plt.axvline(model.best_epoch, linestyle = 'dashed', color = 'blue', label = 'Best val loss: ep ' + str(model.best_epoch))
    plt.title("Loss")
    ax1.legend(frameon=False);

    # Accuracy plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(space, model.train_acc_history, label='Training', color = 'black')
    ax2.plot(space, model.valid_acc_history, label='Validation', color = 'blue')
    plt.xticks(x_ticks)
    plt.axhline(0.99, linestyle = 'dashed', color = 'grey')
    plt.axvline(model.best_epoch, linestyle = 'dashed', color = 'green', label = 'Best val acc: ep ' + str(model.best_epoch))
    plt.title("Accuracy")
    ax2.legend(frameon=False);
