#ifndef COUNTPANEL_H
#define COUNTPANEL_H

#include <QMainWindow>
#include <QListWidget>
#include <QTableWidget>
#include <QSplitter>
#include <QRadioButton>
#include <QTimer>
#include <QFile>
#include "Utilities/paneldialog.h"
#include "Utilities/directorysetter.h"
#include "Utilities/numbertextbox.h"
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
    QCheckBox *chkShow;
    QPushButton *btnColor;
    int obj_id;
    QColor color;
    bool show = false;

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
    void addNewLine(int obj_id);
    QString getTimestampFilename() const;

    QStringList names;
    QListWidget *list;
    QTableWidget *table;
    QSplitter *hSplit;
    DirectorySetter *dirSetter;
    QRadioButton *saveEveryFrame;
    QRadioButton *saveOnInterval;
    NumberTextBox *txtInterval;
    QCheckBox *saveOn;
    QTimer *timer;
    QFile *file = nullptr;
    Darknet *darknet;

    QString headerKey = "CountPanel/header";
    QString hSplitKey = "CountPanel/hSplit";
    QString dirKey    = "CountPanel/dir";

    vector<pair<int, int>> sums;
    vector<pair<int, QCheckBox*>> showObjs;
    vector<pair<int, vector<int>>> sizes;

public slots:
    void itemChanged(QListWidgetItem*);
    void itemClicked(QListWidgetItem*);
    void hSplitMoved(int, int);
    void ping(vector<bbox_t>*);
    void headerChanged(int, int, int);
    void setDir(const QString&);
    void saveOnChecked(int);
    void timeout();

};

class CountDialog : public PanelDialog
{
    Q_OBJECT

public:
    CountDialog(QMainWindow *parent);

};

#endif // COUNTPANEL_H
