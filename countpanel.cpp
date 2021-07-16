#include "countpanel.h"
#include "mainwindow.h"

CountPanel::CountPanel(QMainWindow *parent)
{
    mainWindow = parent;
    Darknet *darknet = (Darknet*)MW->filterDialog->panel->getFilterByName("Darknet");
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

    table = new QTableWidget(0, 2);
    QStringList headers;
    headers << tr("Name") << tr("Value");
    table->setHorizontalHeaderLabels(headers);
    table->verticalHeader()->setVisible(false);
    table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    table->setColumnWidth(1, 60);
    table->setColumnWidth(0, 140);

    //QWidget *tablePanel = new QWidget();
    //QHBoxLayout *tableLayout = new QHBoxLayout();
    //tableLayout->addWidget(table);
    //tablePanel->setLayout(tableLayout);
    //table->setColumnWidth(0, tablePanel->width() - 60);
    //table->setColumnWidth(1, 60);

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(list,    0, 0, 1, 1);
    layout->addWidget(table,   0, 1, 1, 1);
    setLayout(layout);

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

    for (const pair<int, int> sum : sums) {
        int row = rowOf(sum.first);
        if (row > -1) {
            QTableWidgetItem *item = new QTableWidgetItem(QString::number(sum.second));
            table->setItem(row, 1, item);
        }
    }
}

void CountPanel::itemChanged(QListWidgetItem *item)
{
    cout << "itemChanged " << item->text().toStdString() << endl;
    if (item->checkState()) {
        table->setRowCount(table->rowCount() + 1);
        table->setItem(table->rowCount()-1, 0, new QTableWidgetItem(item->text(), 0));
        //QTableWidgetItem *tableItem = table->item(0, table->rowCount());
        //tableItem->setText(item->text());
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
