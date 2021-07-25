#include "countpanel.h"
#include "mainwindow.h"

CountPanel::CountPanel(QMainWindow *parent) : Panel(parent)
{
    darknet = (Darknet*)MW->filterDialog->getPanel()->getFilterByName("Darknet");
    connect(darknet, SIGNAL(ping(vector<bbox_t>*)), this, SLOT(ping(vector<bbox_t>*)));

    timer = new QTimer(this);
    connect(this, SIGNAL(timeout()), this, SLOT(timeout()));

    for (int i = 0; i < darknet->obj_names.size(); i++)
        names.push_back(darknet->obj_names[i].c_str());

    list = new QListWidget();
    list->addItems(names);
    for (int i = 0; i < list->count(); i++) {
        list->item(i)->setFlags(list->item(i)->flags() | Qt::ItemIsUserCheckable);
        list->item(i)->setCheckState(Qt::Unchecked);
    }

    table = new QTableWidget(0, 3);
    QStringList headers;
    headers << tr("Name") << tr("Count") << tr("Show");
    table->setHorizontalHeaderLabels(headers);
    //table->horizontalHeader()->setStretchLastSection(false);

    table->verticalHeader()->setVisible(false);

    //table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    if (MW->settings->contains(headerKey)) {
        table->horizontalHeader()->restoreState(MW->settings->value(headerKey).toByteArray());
    }
    /*
    else {
        table->setColumnWidth(0, 60);
        table->setColumnWidth(1, 60);
        table->setColumnWidth(2, 60);
    }
    */

    for (int i = 0; i < names.size(); i++) {
        ObjDrawer objDrawer(mainWindow, i);
        if (MW->settings->contains(objDrawer.getSettingsKey()))
            addNewLine(objDrawer.obj_id);
    }

    connect(table->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(table->horizontalHeader(), SIGNAL(sectionMoved(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(list, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(itemClicked(QListWidgetItem*)));
    connect(list, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(itemChanged(QListWidgetItem*)));

    hSplit = new QSplitter(this);
    hSplit->addWidget(list);
    hSplit->addWidget(table);
    if (MW->settings->contains(hSplitKey))
        hSplit->restoreState(MW->settings->value(hSplitKey).toByteArray());
    connect(hSplit, SIGNAL(splitterMoved(int, int)), this, SLOT(hSplitMoved(int, int)));

    dirSetter = new DirectorySetter(mainWindow, "Save Directory");
    if (MW->settings->contains(dirKey))
        dirSetter->setPath(MW->settings->value(dirKey).toString());
    else
        dirSetter->setPath(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    connect(dirSetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDir(const QString&)));

    QGroupBox *groupBox = new QGroupBox("Set file save parameters");
    saveEveryFrame = new QRadioButton("Save Every Frame");
    saveOnInterval = new QRadioButton("Save on Interval");
    saveOnInterval->setChecked(true);
    txtInterval = new NumberTextBox();
    txtInterval->setText(MW->settings->value(intervalKey, "60").toString());
    connect(txtInterval, SIGNAL(editingFinished()), this, SLOT(intervalEdited()));
    QLabel *lbl01 = new QLabel("(seconds)");
    QGridLayout *groupLayout = new QGridLayout();
    groupLayout->addWidget(saveEveryFrame,  0, 0, 1, 1);
    groupLayout->addWidget(saveOnInterval,  1, 0, 1, 1);
    groupLayout->addWidget(txtInterval,     1, 1, 1, 1);
    groupLayout->addWidget(lbl01,           1, 2, 1, 1);
    groupBox->setLayout(groupLayout);

    saveOn = new QCheckBox("Save On");
    connect(saveOn, SIGNAL(stateChanged(int)), this, SLOT(saveOnChecked(int)));

    QWidget *filePanel = new QWidget(this);
    QGridLayout *fileLayout = new QGridLayout();
    fileLayout->addWidget(dirSetter,   0, 0, 1, 1);
    fileLayout->addWidget(groupBox,    1, 0, 1, 1);
    fileLayout->addWidget(saveOn,      2, 0, 1, 1);
    filePanel->setLayout(fileLayout);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(hSplit,     0, 0, 1, 1);
    layout->addWidget(filePanel,  1, 0, 1, 1);
    setLayout(layout);
}

void CountPanel::autoSave()
{
    if (changed) {
        cout << "CountPanel::saveSettings" << endl;
        MW->settings->setValue(hSplitKey, hSplit->saveState());
        MW->settings->setValue(headerKey, table->horizontalHeader()->saveState());
        changed = false;
    }
}

void CountPanel::timeout()
{
    cout << "CountPanel::timeout" << endl;
}

void CountPanel::saveOnChecked(int arg)
{
    cout << "CountPanel::saveOnChecked: " << arg << endl;
    if (arg) {
        int interval = txtInterval->intValue();
        if (interval == 0) {
            QMessageBox::warning(this, "PlayQt", "Save on Interval must be set to continue");
            saveOn->setCheckState(Qt::Unchecked);
            return;
        }

        file = new QFile(getTimestampFilename(), this);
        if (!file->open(QFile::WriteOnly | QFile::Text)) {
            QMessageBox::warning(this, "PlayQt", QString("Unable to open file:\n%1").arg(file->fileName()));
            return;
        }
        QTextStream out(file);
        for (int i = 0; i < table->rowCount(); i++) {
            out << ", " << table->item(i, 0)->text() << ", total, avg, std dev";
        }
    }
    else {
        if (file) {
            file->flush();
            file->close();
            file = nullptr;
        }
    }
}

void CountPanel::intervalEdited()
{
    MW->settings->setValue(intervalKey, txtInterval->text());
}

void CountPanel::setDir(const QString& arg)
{
    cout << "CountPanel::setDir" << endl;
    MW->settings->setValue(dirKey, arg);
}

QString CountPanel::getTimestampFilename() const
{
    QString result = dirSetter->directory;
    return result.append("/").append(QDateTime::currentDateTime().toString("yyyyMMddhhmmss")).append(".txt");
}

void CountPanel::headerChanged(int arg1, int arg2, int arg3)
{
    if (isVisible())
        changed = true;
}

void CountPanel::hSplitMoved(int pos, int index)
{
    if (isVisible())
        changed = true;
}

int CountPanel::indexForSums(int obj_id)
{
    int result = -1;
    for (int i = 0; i < sums.size(); i++) {
        if (sums[i].first == obj_id) {
            result = i;
            break;
        }
    }
    return result;
}

int CountPanel::idFromName(const QString& name)
{
    int result = -1;

    for (int i = 0; i < names.size(); i++) {
        if (name == names[i]) {
            result = i;
            break;
        }
    }

    return result;
}

int CountPanel::rowOf(int obj_id)
{
    int result = -1;
    for (int i = 0; i < table->rowCount(); i++) {
        if (table->item(i, 0)->text() == names[obj_id]) {
            result = i;
            break;
        }
    }
    return result;
}

void CountPanel::ping(vector<bbox_t> *detections)
{
    sums.clear();
    for (const bbox_t detection : *detections) {
        int obj_id = detection.obj_id;

        int indexSum = indexForSums(obj_id);
        if (indexSum < 0) {
            sums.push_back(make_pair(obj_id, 1));
        }
        else {
            sums[indexSum].second++;
        }
    }

    for (const pair<int, int>& sum : sums) {
        int row = rowOf(sum.first);
        if (row > -1)
            table->item(row, 1)->setText(QString::number(sum.second));
    }

    for (int i = 0; i < table->rowCount(); i++) {
        if (indexForSums(idFromName(table->item(i, 0)->text())) < 0)
            table->item(i, 1)->setText("0");
    }
}

void CountPanel::addNewLine(int obj_id)
{
    ObjDrawer *objDrawer = new ObjDrawer(mainWindow, obj_id);
    table->setRowCount(table->rowCount() + 1);
    table->setRowHeight(table->rowCount()-1, 17);
    QTableWidgetItem *name = new QTableWidgetItem(names[objDrawer->obj_id]);
    name->setFlags(name->flags() & ~Qt::ItemIsEditable);
    table->setItem(table->rowCount()-1, 0, name);
    QTableWidgetItem *sum = new QTableWidgetItem("0");
    sum->setTextAlignment(Qt::AlignRight);
    sum->setFlags(sum->flags() & ~Qt::ItemIsEditable);
    table->setItem(table->rowCount()-1, 1, sum);
    if (MW->settings->contains(objDrawer->getSettingsKey()))
        objDrawer->restoreState(MW->settings->value(objDrawer->getSettingsKey()).toString());
    list->item(objDrawer->obj_id)->setCheckState(Qt::Checked);
    connect(objDrawer, SIGNAL(shown(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
    connect(objDrawer, SIGNAL(colored(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
    table->setCellWidget(table->rowCount()-1, 2, objDrawer);
    if (objDrawer->show)
        darknet->obj_drawn[objDrawer->obj_id] = objDrawer->color;
    MW->settings->setValue(objDrawer->getSettingsKey(), objDrawer->saveState());
}

void CountPanel::itemChanged(QListWidgetItem *item)
{
    cout << "CountPanel::itemChanged" << endl;
    QString name = item->text();
    int obj_id = idFromName(name);

    if (item->checkState()) {
        addNewLine(obj_id);
    }
    else {
        int index = rowOf(obj_id);
        ObjDrawer *objDrawer = (ObjDrawer*)table->cellWidget(index, 2);
        objDrawer->emit shown(obj_id, YUVColor());
        MW->settings->remove(objDrawer->getSettingsKey());
        table->removeCellWidget(index, 2);
        table->removeRow(index);
    }
}

void CountPanel::itemClicked(QListWidgetItem *item)
{
    if (item->checkState())
            item->setCheckState(Qt::Unchecked);
    else
        item->setCheckState(Qt::Checked);
}

ObjDrawer::ObjDrawer(QMainWindow *parent, int obj_id)
{
    mainWindow = parent;
    this->obj_id = obj_id;
    color = Qt::green;

    chkShow = new QCheckBox();
    btnColor = new QPushButton();
    btnColor->setStyleSheet(getButtonStyle());
    btnColor->setMaximumWidth(15);
    btnColor->setMaximumHeight(15);
    btnColor->setCursor(Qt::PointingHandCursor);
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(chkShow, 0, 0, 1, 1, Qt::AlignRight);
    layout->addWidget(btnColor,   0, 1, 1, 1, Qt::AlignRight);
    layout->setContentsMargins(10, 0, 10, 0);
    setLayout(layout);

    connect(chkShow, SIGNAL(stateChanged(int)), this, SLOT(stateChanged(int)));
    connect(btnColor, SIGNAL(clicked()), this, SLOT(buttonPressed()));
}

QString ObjDrawer::saveState() const
{
    QStringList result;
    result << QString::number(obj_id) << color.name() << QString::number(show);
    QString str = result.join(seperator);
    return str;
}

void ObjDrawer::restoreState(const QString& arg)
{
    QStringList result = arg.split(seperator);

    obj_id = result[0].toInt();
    color = QColor(result[1]);
    if (result[2].toInt())
        show = true;
    else
        show = false;

    chkShow->setChecked(show);
    btnColor->setStyleSheet(getButtonStyle());
}

QString ObjDrawer::getSettingsKey() const
{
    return QString("ObjDrawer_%1").arg(obj_id);
}

QString ObjDrawer::getButtonStyle() const
{
    return QString("QPushButton {background-color: %1;}").arg(color.name());
}

void ObjDrawer::stateChanged(int state)
{
    show = (bool)state;
    MW->settings->setValue(getSettingsKey(), saveState());

    if (state)
        emit shown(obj_id, YUVColor(color));
    else
        emit shown(obj_id, YUVColor());
}

void ObjDrawer::buttonPressed()
{
    color = QColorDialog::getColor(color, MW->countDialog, "PlayQt");
    btnColor->setStyleSheet(getButtonStyle());
    MW->settings->setValue(getSettingsKey(), saveState());

    if (show)
        emit colored(obj_id, YUVColor(color));
}

CountDialog::CountDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle("Counter");
    panel = new CountPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 600;
    defaultHeight = 400;
    settingsKey = "CountDialog/geometry";
}

/*
for (const pair<int, vector<int>>& size : sizes) {
    int row = rowOf(size.first);
    if (row > -1) {
        int count = size.second.size();
        QTableWidgetItem *item = new QTableWidgetItem(QString::number(count));
        table->setItem(row, 1, item);
        double sum = accumulate(begin(size.second), end(size.second), 0.0);
        double mean = sum / count;
        double accum = 0.0;
        for_each(begin(size.second), end(size.second), [&](const double d) {
            accum += (d - mean) * (d - mean);
        });
        double stdev = sqrt(accum / (count - 1));
        QTableWidgetItem *avgItem = new QTableWidgetItem(QString::number(mean));
        table->setItem(row, 2, avgItem);
        QTableWidgetItem *stdItem = new QTableWidgetItem(QString::number(stdev));
        table->setItem(row, 3, stdItem);
    }
}
*/
