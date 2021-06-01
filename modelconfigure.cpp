#include "modelconfigure.h"
#include "mainwindow.h"

ModelConfigureDialog::ModelConfigureDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle("Configure");
    model_configure = new ModelConfigure(parent);
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(model_configure);
    setLayout(layout);
}

int ModelConfigureDialog::getDefaultWidth()
{
    return defaultWidth;
}

int ModelConfigureDialog::getDefaultHeight()
{
    return defaultHeight;
}

ModelConfigure::ModelConfigure(QMainWindow * parent) : QWidget(parent)
{
    mainWindow = parent;

    QPushButton *test = new QPushButton("Test");
    namesFile = new FileSetter(mainWindow, "Names", "Names(*.names)");
    namesFile->trimHeight();
    cfgFile = new FileSetter(mainWindow, "Config", "Config(*.cfg)");
    cfgFile->trimHeight();
    weightsFile = new FileSetter(mainWindow, "Weight", "Weights(*.weights)");
    weightsFile->trimHeight();
    initializeModelOnStartup = new QCheckBox("Initialize Model On Startup");
    QPushButton *loadModel = new QPushButton("Load Model");
    QPushButton *clearModel = new QPushButton("Clear Model");
    QPushButton *clearSettings = new QPushButton("Clear Settings");
    QPushButton *close = new QPushButton("Close");

    modelWidth = new NumberTextBox();
    modelWidth->setMaximumWidth(modelWidth->fontMetrics().boundingRect("00000").width() * 1.5);
    QLabel *lbl00 = new QLabel("Width");
    modelHeight = new NumberTextBox();
    modelHeight->setMaximumWidth(modelHeight->fontMetrics().boundingRect("00000").width() * 1.5);
    QLabel *lbl01 = new QLabel("Height");
    setDims = new QPushButton("Set");
    QGroupBox *resolution = new QGroupBox("Model Resolution");
    QGridLayout *rl = new QGridLayout();
    rl->addWidget(lbl00,                    0, 0, 1, 1, Qt::AlignRight);
    rl->addWidget(modelWidth,               0, 1, 1, 1, Qt::AlignLeft);
    rl->addWidget(lbl01,                    0, 2, 1, 1, Qt::AlignRight);
    rl->addWidget(modelHeight,              0, 3, 1, 1, Qt::AlignLeft);
    rl->addWidget(setDims,                  0, 5, 1, 1, Qt::AlignRight);
    resolution->setLayout(rl);

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(namesFile,                1, 0, 1, 6);
    layout->addWidget(weightsFile,              2, 0, 1, 6);
    layout->addWidget(cfgFile,                  3, 0, 1, 6);
    layout->addWidget(resolution,               4, 0, 1, 6);
    layout->addWidget(initializeModelOnStartup, 6, 0, 1, 2);
    layout->addWidget(loadModel,                7, 0, 1, 1);
    layout->addWidget(clearModel,               7, 1, 1, 1);
    layout->addWidget(clearSettings,            7, 3, 1, 1);
    layout->addWidget(close,                    7, 5, 1, 1);

    setLayout(layout);

    namesFile->setPath(MW->names_file);
    cfgFile->setPath(MW->cfg_file);

    QSize dims = getModelDimensions();
    if (dims != QSize(0,0)) {
        modelWidth->setIntValue(dims.width());
        modelHeight->setIntValue(dims.height());
    }
    setDims->setEnabled(false);

    weightsFile->setPath(MW->weights_file);
    initializeModelOnStartup->setChecked(MW->initializeModelOnStartup);

    connect(test, SIGNAL(clicked()), this, SLOT(test()));
    connect(namesFile, SIGNAL(fileSet(QString)), this, SLOT(setNamesFile(QString)));
    connect(cfgFile, SIGNAL(fileSet(QString)), this, SLOT(setCfgFile(QString)));
    connect(weightsFile, SIGNAL(fileSet(QString)), this, SLOT(setWeightsFile(QString)));
    connect(initializeModelOnStartup, SIGNAL(stateChanged(int)), this, SLOT(setInitializeModelOnStartup(int)));
    connect(loadModel, SIGNAL(clicked()), this, SLOT(loadModel()));
    connect(clearModel, SIGNAL(clicked()), this, SLOT(clearModel()));
    connect(setDims, SIGNAL(clicked()), this, SLOT(setModelDimensions()));
    connect(modelWidth, SIGNAL(textEdited(const QString&)), this, SLOT(cfgEdited(const QString&)));
    connect(modelHeight, SIGNAL(textEdited(const QString&)), this, SLOT(cfgEdited(const QString&)));
    connect(clearSettings, SIGNAL(clicked()), this, SLOT(clearSettings()));
    connect(close, SIGNAL(clicked()), this, SLOT(close()));
}


void ModelConfigure::test()
{
}

void ModelConfigure::clearModel()
{
    delete MW->model;
    MW->model = nullptr;
}

void ModelConfigure::loadModel()
{
    if (MW->model != nullptr) {
        delete MW->model;
    }
    MW->model = new Model(MW);
    MW->model->initialize(MW->cfg_file, MW->weights_file, MW->names_file, 0);
}

QSize ModelConfigure::getModelDimensions()
{
    QFile file(MW->cfg_file);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
        return QSize(0, 0);

    QString contents = file.readAll();
    QString width_key = "width=";
    int width_index = contents.indexOf(width_key);
    int width_length = contents.indexOf("\n", width_index) - (width_index + width_key.length());
    QString arg_width = contents.mid(width_index + width_key.length(), width_length);

    QString height_key = "height=";
    int height_index = contents.indexOf(height_key);
    int height_length = contents.indexOf("\n", height_index) - (height_index + height_key.length());
    QString arg_height = contents.mid(height_index + height_key.length(), height_length);

    return QSize(arg_width.toInt(), arg_height.toInt());
}

void ModelConfigure::setModelDimensions()
{
    int width = modelWidth->text().toInt();
    int height = modelHeight->text().toInt();

    if (width%32 != 0 || height%32 != 0) {
        QMessageBox::critical(MW, "Model Dimension Error", "Model Dimensions must be divisible by 32");
        return;
    }

    modelDimensions = QSize(width, height);

    QFile file(MW->cfg_file);
    if (!file.open(QIODevice::ReadWrite | QIODevice::Text))
        return;

    QString arg_width = "width=";
    arg_width.append(QString::number(width));
    QString arg_height = "height=";
    arg_height.append(QString::number(height));

    QString contents = file.readAll();

    int width_index = contents.indexOf("width");
    int width_length = contents.indexOf("\n", width_index) - width_index;

    contents.remove(width_index, width_length);
    contents.insert(width_index, arg_width);

    int height_index = contents.indexOf("height");
    int height_length = contents.indexOf("\n", height_index) - height_index;
    contents.remove(height_index, height_length);
    contents.insert(height_index, arg_height);

    file.seek(0);
    file.write(contents.toUtf8());
    file.close();
    setDims->setEnabled(false);

}

void ModelConfigure::cfgEdited(const QString &text)
{
    cout << text.toStdString() << endl;
    setDims->setEnabled(true);
}

void ModelConfigure::setNamesFile(QString path)
{
    MW->names_file = path;
    MW->getNames(path);
}

void ModelConfigure::setCfgFile(QString path)
{
    MW->cfg_file = path;
    QSize dims = getModelDimensions();
    if (dims == QSize(0,0)) {
        modelWidth->setText("");
        modelHeight->setText("");
    }
    else{
        modelWidth->setIntValue(dims.width());
        modelHeight->setIntValue(dims.height());
    }
    setDims->setEnabled(false);
}

void ModelConfigure::setWeightsFile(QString path)
{
    MW->weights_file = path;
}

void ModelConfigure::setInitializeModelOnStartup(int arg)
{
    MW->initializeModelOnStartup = arg;
}

void ModelConfigure::clearSettings()
{
    QMessageBox::StandardButton reply = QMessageBox::question(MW, "Clear Settings", "Are you sure you want to clear the settings?");
    if (reply == QMessageBox::Yes) {
        cfgFile->setPath("");
        namesFile->setPath("");
        weightsFile->setPath("");
        setNamesFile("");
        setCfgFile("");
        setWeightsFile("");
    }
}

void ModelConfigure::close()
{
    MW->modelConfigureDialog->hide();
}
