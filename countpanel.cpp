#include "countpanel.h"
#include "mainwindow.h"

CountPanel::CountPanel(QMainWindow *parent) : Panel(parent)
{
    darknet = (Darknet*)MW->filter()->getFilterByName("Darknet");
    connect(darknet, SIGNAL(send(vector<bbox_t>*)), this, SLOT(feed(vector<bbox_t>*)));
    connect(darknet, SIGNAL(namesSet()), this, SLOT(setNames()));

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));

    list = new QListWidget();
    setNames();

    table = new QTableWidget(0, 4);
    table->horizontalHeader()->setStyleSheet(QString("QHeaderView::section:last {border-right: 1px solid %1;}").arg(MW->config()->fd->color.name()));
    QStringList headers;
    headers << tr("Name") << tr("Count") << tr("Show") << tr("Alarm");
    table->setHorizontalHeaderLabels(headers);
    table->verticalHeader()->setVisible(false);

    if (MW->settings->contains(headerKey)) {
        table->horizontalHeader()->restoreState(MW->settings->value(headerKey).toByteArray());
    }
    else {
        table->setColumnWidth(0, 60);
        table->setColumnWidth(1, 60);
        table->setColumnWidth(2, 60);
        table->setColumnWidth(3, 60);
    }

    for (int i = 0; i < names.size(); i++) {
        ObjDrawer objDrawer(mainWindow, i);
        if (MW->settings->contains(objDrawer.getSettingsKey()))
            addNewLine(objDrawer.obj_id);
    }

    connect(table->horizontalHeader(), SIGNAL(sectionResized(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(table->horizontalHeader(), SIGNAL(sectionMoved(int, int, int)), this, SLOT(headerChanged(int, int, int)));
    connect(list, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(itemClicked(QListWidgetItem*)));
    //connect(list, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(itemChanged(QListWidgetItem*)));

    hSplit = new QSplitter(this);
    hSplit->addWidget(list);
    hSplit->addWidget(table);
    if (MW->settings->contains(hSplitKey)) {
        hSplit->restoreState(MW->settings->value(hSplitKey).toByteArray());
    }
    else {
        QList<int> splitSizes;
        splitSizes << 105 << 220;
        hSplit->setSizes(splitSizes);
    }
    connect(hSplit, SIGNAL(splitterMoved(int, int)), this, SLOT(hSplitMoved(int, int)));

    dirSetter = new DirectorySetter(mainWindow, "Save Directory");
    if (MW->settings->contains(dirKey))
        dirSetter->setPath(MW->settings->value(dirKey).toString());
    else
        dirSetter->setPath(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    connect(dirSetter, SIGNAL(directorySet(const QString&)), this, SLOT(setDir(const QString&)));

    txtInterval = new NumberTextBox();
    txtInterval->setMaximumWidth(txtInterval->fontMetrics().boundingRect("0000000").width());
    txtInterval->setText(MW->settings->value(intervalKey, "60").toString());
    connect(txtInterval, SIGNAL(editingFinished()), this, SLOT(intervalEdited()));
    QLabel *lbl00 = new QLabel("Set Save Interval");
    QLabel *lbl01 = new QLabel("(seconds)");
    QGridLayout *intervalLayout = new QGridLayout();
    intervalLayout->addWidget(lbl00,         0, 0, 1, 1);
    intervalLayout->addWidget(txtInterval,   0, 1, 1, 1);
    intervalLayout->addWidget(lbl01,         0, 2, 1, 1);
    intervalLayout->addWidget(new QLabel,    0, 3, 1, 1);
    intervalLayout->setColumnStretch(3, 10);
    intervalLayout->setContentsMargins(0, 0, 0, 0);
    intervalPanel = new QWidget(this);
    intervalPanel->setLayout(intervalLayout);

    saveOn = new QCheckBox("Save On");
    saveOn->setChecked(MW->settings->value(saveOnKey, false).toBool());
    connect(saveOn, SIGNAL(clicked(bool)), this, SLOT(saveOnClicked(bool)));

    QWidget *filePanel = new QWidget(this);
    QGridLayout *fileLayout = new QGridLayout();
    fileLayout->addWidget(dirSetter,      0, 0, 1, 2);
    fileLayout->addWidget(intervalPanel,  1, 0, 1, 1);
    fileLayout->addWidget(saveOn,         1, 1, 1, 1);
    filePanel->setLayout(fileLayout);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(hSplit,     0, 0, 1, 1);
    layout->addWidget(filePanel,  1, 0, 1, 1);
    layout->setRowStretch(0, 10);
    setLayout(layout);

    connect(MW->control(), SIGNAL(quitting()), this, SLOT(shutdown()));

}

CountPanel::~CountPanel()
{
    if (file) {
        file->flush();
        file->close();
    }
}

void CountPanel::setNames() {

    cout << "CountPanel::setNames" << endl;

    for (int i = 0; i < list->count(); i++) {
        if (list->item(i)->checkState() == Qt::Checked) {
            int obj_id = idFromName(list->item(i)->text());
            removeLine(obj_id);
        }
    }

    list->clear();
    names.clear();

    for (int i = 0; i < darknet->obj_names.size(); i++)
        names.push_back(darknet->obj_names[i].c_str());

    list->addItems(names);
    for (int i = 0; i < list->count(); i++) {
        list->item(i)->setFlags(list->item(i)->flags() | Qt::ItemIsUserCheckable);
        list->item(i)->setCheckState(Qt::Unchecked);
    }
}

void CountPanel::autoSave()
{
    if (changed) {
        MW->settings->setValue(hSplitKey, hSplit->saveState());
        MW->settings->setValue(headerKey, table->horizontalHeader()->saveState());
        changed = false;
    }
}

void CountPanel::feed(vector<bbox_t> *detections)
{
    if (saveOn->isChecked() && file == nullptr)
        saveOnClicked(true);

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
        if (row > -1) {
            table->item(row, 1)->setText(QString::number(sum.second));

            int count = sum.second;

            ((AlarmSetter*)table->cellWidget(row, 3))->alarmDialog->getPanel()->feed(count);
            if (saveOn->isChecked())
                addCount(sum.first, sum.second);
        }
    }

    for (int i = 0; i < table->rowCount(); i++) {
        int obj_id = idFromName(table->item(i, 0)->text());
        if (indexForSums(obj_id) < 0) {
            table->item(i, 1)->setText("0");

            ((AlarmSetter*)table->cellWidget(i, 3))->alarmDialog->getPanel()->feed(0);
            if (saveOn->isChecked())
                addCount(obj_id, 0);
        }
    }
}

void CountPanel::timeout()
{
    if (counts.empty())
        return;

    mutex.lock();
    QTextStream out(file);
    out << QTime::currentTime().toString("hh:mm:ss") << ", ";
    out << MW->is->formatTime(MW->is->get_master_clock());
    sort(counts.begin(), counts.end(), [](const pair<int, vector<int>>& left, const pair<int, vector<int>>& right) {
        return left.first < right.first;
    });
    for (pair<int, vector<int>>& count : counts) {
        int row = rowOf(count.first);
        if (row > -1) {
            int samples = count.second.size();
            double sum = accumulate(begin(count.second), end(count.second), 0.0);
            double mean = sum / samples;
            double accum = 0.0;
            for_each(begin(count.second), end(count.second), [&](const double d) {
                accum += (d - mean) * (d - mean);
            });
            double stdev = sqrt(accum / (samples - 1));
            out << ", " << sum << ", " << mean << ", " << stdev;
            count.second.clear();
        }
    }
    out << "\n";
    counts.clear();
    mutex.unlock();
}

void CountPanel::saveOnClicked(bool checked)
{
    if (checked) {
        int interval = txtInterval->intValue();
        if (interval == 0) {
            QMessageBox::warning(this, "playqt", "Save on Interval must be set to continue");
            saveOn->setCheckState(Qt::Unchecked);
            return;
        }

        file = new QFile(getTimestampFilename(), this);
        if (!file->open(QFile::WriteOnly | QFile::Text)) {
            QMessageBox::warning(this, "playqt", QString("Unable to open file:\n%1").arg(file->fileName()));
            return;
        }

        QTextStream out(file);
        out << "system time, stream time";
        vector<int> output_ids;
        for (int i = 0; i < table->rowCount(); i++)
            output_ids.push_back(idFromName(table->item(i, 0)->text()));

        sort(output_ids.begin(), output_ids.end());
        for (int i = 0; i < output_ids.size(); i++)
            out << ", " <<  names[output_ids[i]] << " total, avg, std dev";

        out << "\n";

        list->setEnabled(false);
        dirSetter->setEnabled(false);
        intervalPanel->setEnabled(false);
        timer->start(interval * 1000);
    }
    else {
        shutdown();
    }

    MW->settings->setValue(saveOnKey, checked);
}

void CountPanel::shutdown()
{
    if (timer->isActive())
        timer->stop();
    if (file) {
        file->flush();
        file->close();
        file = nullptr;
    }
    list->setEnabled(true);
    dirSetter->setEnabled(true);
    intervalPanel->setEnabled(true);
}

void CountPanel::intervalEdited()
{
    MW->settings->setValue(intervalKey, txtInterval->text());
}

void CountPanel::setDir(const QString& arg)
{
    MW->settings->setValue(dirKey, arg);
}

QString CountPanel::getTimestampFilename() const
{
    QString result = dirSetter->directory;
    return result.append("/").append(QDateTime::currentDateTime().toString("yyyyMMddhhmmss")).append(".csv");
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

int CountPanel::indexForCounts(int obj_id)
{
    int result = -1;
    for (int i = 0; i < counts.size(); i++) {
        if (counts[i].first == obj_id) {
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

void CountPanel::addCount(int obj_id, int count)
{
    int index = indexForCounts(obj_id);
    mutex.lock();
    if (index < 0) {
        vector<int> counter;
        counter.push_back(count);
        counts.push_back(make_pair(obj_id, counter));
    }
    else {
        counts[index].second.push_back(count);
    }
    mutex.unlock();
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
    sum->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
    sum->setFlags(sum->flags() & ~Qt::ItemIsEditable);
    table->setItem(table->rowCount()-1, 1, sum);
    if (MW->settings->contains(objDrawer->getSettingsKey())) {
        objDrawer->restoreState(MW->settings->value(objDrawer->getSettingsKey()).toString());
    }
    else {
        objDrawer->chkShow->setChecked(true);
        objDrawer->show = true;
    }
    list->item(objDrawer->obj_id)->setCheckState(Qt::Checked);
    connect(objDrawer, SIGNAL(shown(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
    connect(objDrawer, SIGNAL(colored(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
    table->setCellWidget(table->rowCount()-1, 2, objDrawer);
    if (objDrawer->show)
        darknet->obj_drawn[objDrawer->obj_id] = objDrawer->color;
    MW->settings->setValue(objDrawer->getSettingsKey(), objDrawer->saveState());
    AlarmSetter *setter = new AlarmSetter(mainWindow, obj_id);
    table->setCellWidget(table->rowCount()-1, 3, setter);
}

void CountPanel::removeLine(int obj_id)
{
    int index = rowOf(obj_id);
    ObjDrawer *objDrawer = (ObjDrawer*)table->cellWidget(index, 2);
    objDrawer->signalShown(obj_id, YUVColor());
    MW->settings->remove(objDrawer->getSettingsKey());
    table->removeCellWidget(index, 2);
    table->removeRow(index);
}

void CountPanel::itemChanged(QListWidgetItem *item)
{
    QString name = item->text();
    int obj_id = idFromName(name);

    if (item->checkState()) {
        addNewLine(obj_id);
    }
    else {
        removeLine(obj_id);
    }
}

void CountPanel::itemClicked(QListWidgetItem *item)
{
    if (item->checkState())
            item->setCheckState(Qt::Unchecked);
    else
        item->setCheckState(Qt::Checked);

    itemChanged(item);
}

AlarmSetter::AlarmSetter(QMainWindow *parent, int obj_id)
{
    mainWindow = parent;
    this->obj_id = obj_id;
    button = new QPushButton("...");
    button->setMaximumWidth(30);
    button->setMaximumHeight(20);
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(button, 0, 0, 1, 1);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    alarmDialog = new AlarmDialog(mainWindow, obj_id);

    connect(button, SIGNAL(clicked()), this, SLOT(buttonPressed()));
}

void AlarmSetter::buttonPressed()
{
    QString obj_name = MW->count()->names[obj_id];
    alarmDialog->setWindowTitle(QString("Alarm Configuration - ") + obj_name);
    alarmDialog->show();
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

    connect(chkShow, SIGNAL(clicked(bool)), this, SLOT(chkShowClicked(bool)));
    connect(btnColor, SIGNAL(clicked()), this, SLOT(btnColorClicked()));
}

void ObjDrawer::signalShown(int obj_id, const YUVColor& color)
{
    emit shown(obj_id, color);
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

void ObjDrawer::chkShowClicked(bool checked)
{
    show = checked;
    MW->settings->setValue(getSettingsKey(), saveState());

    if (checked)
        emit shown(obj_id, YUVColor(color));
    else
        emit shown(obj_id, YUVColor());
}

void ObjDrawer::btnColorClicked()
{
    QColor result = QColorDialog::getColor(color, MW->countDialog, "playqt");
    if (result.isValid()) {
        color = result;
        btnColor->setStyleSheet(getButtonStyle());
        MW->settings->setValue(getSettingsKey(), saveState());

        if (show)
            emit colored(obj_id, YUVColor(color));
    }
}

CountDialog::CountDialog(QMainWindow *parent) : PanelDialog(parent)
{
    setWindowTitle("Counter");
    panel = new CountPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 410;
    defaultHeight = 280;
    settingsKey = "CountDialog/geometry";
}
