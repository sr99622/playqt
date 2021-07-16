#ifndef COUNTPANEL_H
#define COUNTPANEL_H

#include <QMainWindow>
#include <QListWidget>
#include <QTableWidget>
#include "Utilities/paneldialog.h"
#include "Filters/darknet.h"

class CountPanel : public QWidget
{
    Q_OBJECT

public:
    CountPanel(QMainWindow *parent);
    int indexOf(int obj_id);
    int rowOf(int obj_id);

    QMainWindow *mainWindow;
    QStringList names;
    QListWidget *list;
    QTableWidget *table;

    vector<pair<int, int>> sums;

public slots:
    void itemDoubleClicked(QListWidgetItem*);
    void itemChanged(QListWidgetItem*);
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
