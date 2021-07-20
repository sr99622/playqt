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
    QString getSettingsKey() const;
    QString saveState() const;
    void restoreState(const QString& arg);

    QMainWindow *mainWindow;
    QCheckBox *checkBox;
    QPushButton *button;
    int obj_id;
    QColor color;
    bool show;

    const QString seperator = "\n";

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
    void autoSave() override;
    int indexForSums(int obj_id);
    int rowOf(int obj_id);
    int idFromName(const QString& name);

    QStringList names;
    QListWidget *list;
    QTableWidget *table;
    QSplitter *hSplit;
    Darknet *darknet;

    QString headerKey = "CountPanel/header";
    QString hSplitKey = "CountPanel/hSplit";

    vector<pair<int, int>> sums;
    vector<pair<int, QCheckBox*>> showObjs;
    vector<pair<int, vector<int>>> sizes;

public slots:
    void itemChanged(QListWidgetItem*);
    void itemClicked(QListWidgetItem*);
    void hSplitMoved(int, int);
    void ping(vector<bbox_t>*);
    void headerChanged(int, int, int);

};

class CountDialog : public PanelDialog
{
    Q_OBJECT

public:
    CountDialog(QMainWindow *parent);

};

#endif // COUNTPANEL_H
