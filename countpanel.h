#ifndef COUNTPANEL_H
#define COUNTPANEL_H

#include <QMainWindow>
#include <QListWidget>
#include <QTableWidget>
#include <QSplitter>
#include "Utilities/paneldialog.h"
#include "Filters/darknet.h"

class ObjDrawer : public QWidget
{
    Q_OBJECT

public:
    ObjDrawer(QMainWindow *parent, int obj_id);
    QString getButtonStyle() const;

    QMainWindow *mainWindow;
    QCheckBox *checkBox;
    QPushButton *button;
    int obj_id;
    QColor color;

signals:
    void shown(int, const YUVColor&);
    void colored(int, const YUVColor&);

public slots:
    void stateChanged(int);
    void buttonPressed();

};

class CountPanel : public Panel
{
    Q_OBJECT

public:
    CountPanel(QMainWindow *parent);
    void saveSettings() override;
    int indexForSums(int obj_id);
    int rowOf(int obj_id);
    int idFromName(const QString& name);

    QStringList names;
    QListWidget *list;
    QTableWidget *table;
    QSplitter *hSplit;
    QString hSplitKey = "CountPanel/hSplit";
    Darknet *darknet;

    vector<pair<int, int>> sums;
    vector<pair<int, QCheckBox*>> showObjs;
    vector<pair<int, vector<int>>> sizes;

public slots:
    void itemDoubleClicked(QListWidgetItem*);
    void itemChanged(QListWidgetItem*);
    void itemClicked(QListWidgetItem*);
    void hSplitMoved(int, int);
    void ping(vector<bbox_t>*);

};

class CountDialog : public PanelDialog
{
    Q_OBJECT

public:
    CountDialog(QMainWindow *parent);

    //CountPanel *panel;

};

#endif // COUNTPANEL_H
