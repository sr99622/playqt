#include "darknet.h"
#include "mainwindow.h"

Darknet::Darknet(QMainWindow *parent)
{
    mainWindow = parent;
    name = "Darknet";
    panel = new Panel(mainWindow);

    names = new FileSetter(mainWindow, "Names", "Names(*.names)");
    names->setPath(MW->settings->value(namesKey, "").toString());
    setNames(names->filename);
    connect(names, SIGNAL(fileSet(const QString&)), this, SLOT(setNames(const QString&)));

    weights = new FileSetter(mainWindow, "Weight", "Weights(*.weights)");
    weights->setPath(MW->settings->value(weightsKey, "").toString());
    connect(weights, SIGNAL(fileSet(const QString&)), this, SLOT(setWeights(const QString&)));

    cfg = new FileSetter(mainWindow, "Config", "Config(*.cfg)");
    cfg->setPath(MW->settings->value(cfgKey, "").toString());
    connect(cfg, SIGNAL(fileSet(const QString&)), this, SLOT(setCfg(const QString&)));

    modelWidth = new NumberTextBox();
    modelWidth->setMaximumWidth(modelWidth->fontMetrics().boundingRect("00000").width() * 1.5);
    connect(modelWidth, SIGNAL(textEdited(const QString&)), this, SLOT(cfgEdited(const QString&)));
    QLabel *lbl00 = new QLabel("Width");

    modelHeight = new NumberTextBox();
    modelHeight->setMaximumWidth(modelHeight->fontMetrics().boundingRect("00000").width() * 1.5);
    connect(modelHeight, SIGNAL(textEdited(const QString&)), this, SLOT(cfgEdited(const QString&)));
    QLabel *lbl01 = new QLabel("Height");

    QSize dims = getModelDimensions();
    if (dims != QSize(0,0)) {
        modelWidth->setIntValue(dims.width());
        modelHeight->setIntValue(dims.height());
    }

    setDims = new QPushButton("Set");
    setDims->setMaximumWidth(50);
    connect(setDims, SIGNAL(clicked()), this, SLOT(setModelDimensions()));
    setDims->setEnabled(false);
    QGroupBox *resolution = new QGroupBox("Model Resolution");
    QGridLayout *rl = new QGridLayout();
    rl->addWidget(lbl00,                    0, 0, 1, 1, Qt::AlignRight);
    rl->addWidget(modelWidth,               0, 1, 1, 1, Qt::AlignLeft);
    rl->addWidget(lbl01,                    0, 2, 1, 1, Qt::AlignRight);
    rl->addWidget(modelHeight,              0, 3, 1, 1, Qt::AlignLeft);
    rl->addWidget(setDims,                  0, 5, 1, 1, Qt::AlignRight);
    resolution->setLayout(rl);

    threshold = MW->settings->value(thresholdKey, 0.2f).toFloat();
    sldrThreshold = new QSlider(Qt::Horizontal, mainWindow);
    sldrThreshold->setValue(threshold * 100);
    connect(sldrThreshold, SIGNAL(valueChanged(int)), this, SLOT(setThreshold(int)));
    QLabel *lbl02 = new QLabel("Threshold");
    lblThreshold = new QLabel(QString::number(sldrThreshold->value()));

    QPushButton *loadModel = new QPushButton("Load Model");
    connect(loadModel, SIGNAL(clicked()), this, SLOT(loadModel()));
    QPushButton *clearModel = new QPushButton("Clear Model");
    connect(clearModel, SIGNAL(clicked()), this, SLOT(clearModel()));
    QPushButton *clearSettings = new QPushButton("Clear Settings");
    connect(clearSettings, SIGNAL(clicked()), this, SLOT(clearSettings()));

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(names,                1, 0, 1, 6);
    layout->addWidget(weights,              2, 0, 1, 6);
    layout->addWidget(cfg,                  3, 0, 1, 6);
    layout->addWidget(resolution,           4, 0, 1, 6);
    layout->addWidget(lbl02,                5, 0, 1, 1, Qt::AlignRight);
    layout->addWidget(sldrThreshold,        5, 1, 1, 4);
    layout->addWidget(lblThreshold,         5, 5, 1, 1);
    layout->addWidget(loadModel,            7, 0, 1, 1);
    layout->addWidget(clearModel,           7, 1, 1, 1);
    layout->addWidget(clearSettings,        7, 3, 1, 1);

    panel->setLayout(layout);
}

void Darknet::filter(Frame *vp)
{
    if (model == nullptr) {
        loading = true;
        model = new DarknetModel(mainWindow);
        model->initialize(cfg->filename, weights->filename, names->filename, 0);
    }

    if (!loading) {
        result = model->infer(vp, threshold);
        vector<bbox_t> pinger = result;
        emit ping(&pinger);

        for (size_t i = 0; i < result.size(); i++) {
            if (obj_drawn[result[i].obj_id].isValid()) {
                QRect rect(result[i].x, result[i].y, result[i].w, result[i].h);
                vp->drawBox(rect, 1, obj_drawn[result[i].obj_id]);
            }
        }
    }
}

void Darknet::autoSave()
{
    if (changed) {
        cout << "DarknetPanel::autoSave" << endl;
        MW->settings->setValue(thresholdKey, threshold);
        changed = false;
    }
}

void Darknet::setThreshold(int arg)
{
    lblThreshold->setText(QString::number(arg));
    threshold = arg / (float)100;
    changed = true;
}

void Darknet::clearModel()
{
    if (model != nullptr) {
        if (model->detector != nullptr) {
            model->detector->~Detector();
        }
        delete model;
    }
    model = nullptr;
}

void Darknet::loadModel()
{
    clearModel();
    model = new DarknetModel(mainWindow);
    model->initialize(cfg->filename, weights->filename, names->filename, 0);
}

void Darknet::setModelDimensions()
{
    int width = modelWidth->text().toInt();
    int height = modelHeight->text().toInt();

    if (width%32 != 0 || height%32 != 0) {
        QMessageBox::critical(MW->filterDialog, "Model Dimension Error", "Model Dimensions must be divisible by 32");
        return;
    }

    modelDimensions = QSize(width, height);

    QFile file(cfg->filename);
    if (!file.exists())
        return;

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

QSize Darknet::getModelDimensions()
{
    QFile file(cfg->filename);

    if (!file.exists())
        return QSize(0, 0);

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

void Darknet::clearSettings()
{
    QMessageBox::StandardButton reply = QMessageBox::question(MW->filterDialog, "Clear Settings", "Are you sure you want to clear the settings?");
    if (reply == QMessageBox::Yes) {
        cfg->setPath("");
        names->setPath("");
        weights->setPath("");
        setNames("");
        setCfg("");
        setWeights("");
    }
}

void Darknet::setNames(const QString &path)
{
    MW->settings->setValue(namesKey, path);

    ifstream file(path.toStdString());
    if (!file.is_open()) {
        QString str;
        QTextStream(&str) << "Unable to load model names file: " << path;
        MW->msg(str);
        return;
    }

    obj_names.clear();
    obj_drawn.clear();
    for (string line; getline(file, line);) {
        obj_names.push_back(line);
        obj_drawn.push_back(YUVColor());
    }
}

void Darknet::cfgEdited(const QString &text)
{
    setDims->setEnabled(true);
}

void Darknet::setCfg(const QString &path)
{
    QSize dims = getModelDimensions();
    if (dims == QSize(0, 0)) {
        modelWidth->setText("");
        modelHeight->setText("");
    }
    else {
        modelWidth->setIntValue(dims.width());
        modelHeight->setIntValue(dims.height());
    }
    setDims->setEnabled(false);
    MW->settings->setValue(cfgKey, path);
}

void Darknet::setWeights(const QString &path)
{
    MW->settings->setValue(weightsKey, path);
}

DarknetLoader::DarknetLoader(QObject *model)
{
    this->model = model;
}

void DarknetLoader::run()
{
    ((DarknetModel*)model)->detector = new Detector(cfg_file, weights_file, gpu_id);
    emit done(0);
    MainWindow *mainWindow = (MainWindow*)((DarknetModel*)model)->mainWindow;
    FilterPanel *panel = mainWindow->filter();
    ((Darknet*)panel->getFilterByName("Darknet"))->loading = false;
}

DarknetModel::DarknetModel(QMainWindow *parent) : QObject(parent)
{
    mainWindow = parent;
    waitBox = new WaitBox(mainWindow);
    loader = new DarknetLoader(this);
    connect(loader, SIGNAL(done(int)), waitBox, SLOT(done(int)));
    connect(this, SIGNAL(msg(const QString&)), mainWindow, SLOT(msg(const QString&)));
}

void DarknetModel::initialize(QString cfg_file, QString weights_file, QString names_file, int gpu_id)
{
    if (detector != nullptr)
        detector->~Detector();

    if (!QFile::exists(cfg_file)) {
        QString str;
        QTextStream(&str) << "Unable to load config file: " << cfg_file;
        QMessageBox::critical(MW, "Model Config Load Error", str);
        return;
    }

    if (!QFile::exists(weights_file)) {
        QString str;
        QTextStream(&str) << "Unable to load weights file: " << weights_file;
        QMessageBox::critical(MW, "Model Weights Load Error", str);
        return;
    }

    ifstream file(names_file.toLatin1().data());
    if (!file.is_open()) {
        QString str;
        QTextStream(&str) << "Unable to load model names file: " << names_file;
        QMessageBox::critical(MW, "Model Names Load Error", str);
        return;
    }

    loader->cfg_file = cfg_file.toStdString();
    loader->weights_file = weights_file.toStdString();
    loader->gpu_id = gpu_id;
    QThreadPool::globalInstance()->tryStart(loader);
    waitBox->exec();
}

vector<bbox_t> DarknetModel::infer(Frame *vp, float detection_threshold)
{
    vector<bbox_t> result;

    try {
        //image_t img = hw_get_image(vp);
        image_t img = get_image(vp);
        result = detector->detect(img, detection_threshold);
        detector->free_image(img);
    }
    catch(const exception &e) {
        cout << "Model::infer error: " << e.what() << endl;
        QString error_msg = QString("Model::infer error: %1").arg(e.what());
        emit msg(error_msg);
    }

    return result;
}

image_t DarknetModel::get_image(Frame *vp)
{
    image_t img;

    img.h = vp->frame->height;
    img.w = vp->frame->linesize[0];
    //img.w = vp->frame->width;
    img.c = 3;
    img.data = (float*)malloc(sizeof(float) * img.w * img.h * 3);
    memset(img.data, 0, sizeof(float) * img.w * img.h * 3);

    for (int y = 0; y < img.h; y++) {
        for (int x = 0; x < img.w; x++) {
            int i = y * img.w + x;
            img.data[i] = (float)vp->frame->data[0][i] / 255.0f;
        }
    }

    return img;
}

image_t DarknetModel::hw_get_image(Frame *vp)
{
    Npp8u *pSrc;
    Npp32f *pNorm;
    image_t img;
    img.h = vp->frame->height;
    img.w = vp->frame->linesize[0];
    img.c = 3;
    img.data = (float*)malloc(sizeof(float) * img.w * img.h * 3);
    int frame_size = img.w * img.h;

    try {
        eh.ck(cudaMalloc((void**)(& pSrc), sizeof(Npp8u) * frame_size), "Allocate device source buffer");
        eh.ck(cudaMemcpy(pSrc, vp->frame->data[0], sizeof(Npp8u) * frame_size, cudaMemcpyHostToDevice), "Copy frame picture data buffer to device");
        eh.ck(cudaMalloc((void **)(&pNorm), sizeof(Npp32f) * frame_size), "Allocate a float buffer version of source for device");
        eh.ck(nppsConvert_8u32f(pSrc, pNorm, frame_size), "Convert frame data to float");
        eh.ck(nppsDivC_32f_I(255.0f, pNorm, frame_size), "normalize frame data");
        eh.ck(cudaMemcpy(img.data, pNorm, sizeof(Npp32f) * frame_size, cudaMemcpyDeviceToHost), "copy normalized frame data to img");
        eh.ck(cudaFree(pNorm));;
        eh.ck(cudaFree(pSrc));
    }
    catch (const exception& e) {
        cout << e.what() << endl;
    }

    return img;
}

QString Darknet::getName(int obj_id) const
{
    QString result;
    if (obj_names.size() > obj_id)
        result = obj_names[obj_id].c_str();
    return result;
}

void Darknet::draw(int obj_id, const YUVColor& color)
{
    cout << "DarknetModel::draw" << endl;
    if (obj_drawn.size() > obj_id)
        obj_drawn[obj_id] = color;
}

void DarknetModel::show_console_result(vector<bbox_t> const result_vec, vector<string> const obj_names, int frame_id)
{
    if (frame_id >= 0) cout << " Frame: " << frame_id << endl;
    for (auto &i : result_vec) {
        if (obj_names.size() > i.obj_id) cout << obj_names[i.obj_id] << " - ";

        cout << "obj_id = " << i.obj_id << ", x = " << i.x << ", y = " << i.y
             << ", w = " << i.w << ", h = " << i.h
             << ", prob = " << i.prob << endl;

    }
}
