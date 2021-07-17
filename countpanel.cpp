#include "countpanel.h"
#include "mainwindow.h"

CountPanel::CountPanel(QMainWindow *parent)
{
    mainWindow = parent;
    darknet = (Darknet*)MW->filterDialog->panel->getFilterByName("Darknet");
    ifstream file(darknet->names->filename.toLatin1().data());
    if (file.is_open()) {
        names.clear();
        for (string line; getline(file, line);)
            names.push_back(line.c_str());
    }
    list = new QListWidget();
    list->addItems(names);
    for (int i = 0; i < list->count(); i++) {
        list->item(i)->setFlags(list->item(i)->flags() | Qt::ItemIsUserCheckable);
        list->item(i)->setCheckState(Qt::Unchecked);
    }

    table = new QTableWidget(0, 3);
    QStringList headers;
    headers << tr("Name") << tr("Value") << tr("Show");
    table->setHorizontalHeaderLabels(headers);
    table->verticalHeader()->setVisible(false);
    table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    table->setColumnWidth(1, 60);
    table->setColumnWidth(0, 140);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(list,    0, 0, 1, 1);
    layout->addWidget(table,   0, 1, 1, 1);
    setLayout(layout);

    connect(list, SIGNAL(itemClicked(QListWidgetItem*)), this, SLOT(itemClicked(QListWidgetItem*)));
    connect(list, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(itemDoubleClicked(QListWidgetItem*)));
    connect(list, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(itemChanged(QListWidgetItem*)));
    connect(darknet, SIGNAL(ping(vector<bbox_t>*)), this, SLOT(ping(vector<bbox_t>*)));
}

int CountPanel::indexOf(int obj_id)
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
        QTableWidgetItem *item = table->item(i, 0);
        if (item->text() == names[obj_id]) {
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

        int index = indexOf(obj_id);
        if (index < 0) {
            sums.push_back(make_pair(obj_id, 1));
        }
        else {
            sums[index].second++;
        }
    }

    for (const pair<int, int>& sum : sums) {
        int row = rowOf(sum.first);
        if (row > -1) {
            QTableWidgetItem *item = new QTableWidgetItem(QString::number(sum.second));
            table->setItem(row, 1, item);
        }
    }
}

void CountPanel::itemChanged(QListWidgetItem *item)
{
    QString name = item->text();
    int obj_id = idFromName(name);

    cout << "itemChanged name: " << name.toStdString() << " obj_id: " << obj_id << endl;

    if (item->checkState()) {
        table->setRowCount(table->rowCount() + 1);
        table->setItem(table->rowCount()-1, 0, new QTableWidgetItem(name, 0));
        ObjDrawer *objDrawer = new ObjDrawer(mainWindow, obj_id);
        connect(objDrawer, SIGNAL(shown(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
        connect(objDrawer, SIGNAL(colored(int, const YUVColor&)), darknet, SLOT(draw(int, const YUVColor&)));
        table->setCellWidget(table->rowCount()-1, 2, objDrawer);
    }
    else {
        int index = rowOf(obj_id);
        ObjDrawer *objDrawer = (ObjDrawer*)table->cellWidget(index, 2);
        objDrawer->emit shown(obj_id, YUVColor());
        table->removeCellWidget(index, 2);
        table->removeRow(index);
    }
}

void CountPanel::itemDoubleClicked(QListWidgetItem *item)
{
    if (item->checkState())
            item->setCheckState(Qt::Unchecked);
    else
        item->setCheckState(Qt::Checked);
    cout << "itemDoubleClicked " << item->text().toStdString() << endl;
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

    checkBox = new QCheckBox("show");
    button = new QPushButton();
    button->setStyleSheet(getButtonStyle());
    button->setMaximumWidth(button->fontMetrics().boundingRect("XXX").width() * 1.5);
    QHBoxLayout *layout = new QHBoxLayout();
    layout->addWidget(checkBox);
    layout->addWidget(button);
    layout->setContentsMargins(0, 0, 0, 0);
    setLayout(layout);

    connect(checkBox, SIGNAL(stateChanged(int)), this, SLOT(stateChanged(int)));
    connect(button, SIGNAL(clicked()), this, SLOT(buttonPressed()));
}

QString ObjDrawer::getButtonStyle() const
{
    return QString("QPushButton {background-color: %1;}").arg(color.name());
}

void ObjDrawer::stateChanged(int state)
{
    cout << "ObjDrawer::stateChanged: " << obj_id << endl;

    if (state)
        emit shown(obj_id, YUVColor(color));
    else
        emit shown(obj_id, YUVColor());
}

void ObjDrawer::buttonPressed()
{
    cout << "ObjDrawer::buttonPressed: " << obj_id << endl;

    color = QColorDialog::getColor(color, MW->countDialog, "PlayQt");
    button->setStyleSheet(getButtonStyle());

    emit colored(obj_id, YUVColor(color));
}

CountDialog::CountDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Counter");
    panel = new CountPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);
}

int CountDialog::getDefaultWidth()
{
    return defaultWidth;
}

int CountDialog::getDefaultHeight()
{
    return defaultHeight;
}

QString CountDialog::getSettingsKey() const
{
    return settingsKey;
}
