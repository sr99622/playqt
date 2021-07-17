#ifndef COUNTPANEL_H
#define COUNTPANEL_H

#include <QMainWindow>
#include <QListWidget>
#include <QTableWidget>
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

class CountPanel : public QWidget
{
    Q_OBJECT

public:
    CountPanel(QMainWindow *parent);
    int indexOf(int obj_id);
    int rowOf(int obj_id);
    int idFromName(const QString& name);

    QMainWindow *mainWindow;
    QStringList names;
    QListWidget *list;
    QTableWidget *table;
    Darknet *darknet;

    vector<pair<int, int>> sums;
    vector<pair<int, QCheckBox*>> showObjs;

public slots:
    void itemDoubleClicked(QListWidgetItem*);
    void itemChanged(QListWidgetItem*);
    void itemClicked(QListWidgetItem*);
    void ping(vector<bbox_t>*);

};

class CountDialog : public PanelDialog
{
    Q_OBJECT

public:
    CountDialog(QMainWindow *parent);

    QMainWindow *mainWindow;
    CountPanel *panel;

    int getDefaultWidth() override;
    int getDefaultHeight() override;
    QString getSettingsKey() const override;

    const int defaultWidth = 520;
    const int defaultHeight = 600;
    const QString settingsKey = "CountDialog/geometry";

};

#endif // COUNTPANEL_H
