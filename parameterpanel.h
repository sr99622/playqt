#ifndef PARAMETERPANEL_H
#define PARAMETERPANEL_H

#include <QMainWindow>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QSettings>
#include <QCheckBox>

#include "Utilities/paneldialog.h"
#include "Ffplay/CommandOptions.h"

using namespace std;

class OptionBox : public QComboBox
{
public:
    OptionBox(QWidget *parent = nullptr);
    bool keyInput = false;

protected:
    void keyPressEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent *event);
};

class StoredOption : public QListWidgetItem
{
public:
    StoredOption(const QString& text);
    const QString toString();
    QString arg;
};

class SavedCmdLines : public QListWidget
{

public:
    SavedCmdLines(QMainWindow *parent);
    QMainWindow *mainWindow;
    void keyPressEvent(QKeyEvent *event) override;

};

class ParameterPanel : public QWidget
{
    Q_OBJECT

public:
    ParameterPanel(QMainWindow *parent);
    void setCmdLine();
    void addOptionToSaver(OptionDef option);
    const QString getOptionStorageString();
    void saveSettings(QSettings *settings);
    void restoreSettings(QSettings *settings);
    void set(int option_index, char * arg);

    QMainWindow *mainWindow;
    OptionBox *options;
    QLineEdit *parameter;
    QLabel *cmdLineEquiv;
    QLineEdit *cmdLineName;
    SavedCmdLines *savedCmdLines;
    vector<OptionDef> saved_options;

public slots:
    void set();
    void clear();
    void optionChanged(int);
    void parameterEntered();
    void saveCmdLine();
    void itemDoubleClicked(QListWidgetItem*);
    void clearSavedCmdLines();
};

class ParameterDialog : public PanelDialog
{
    Q_OBJECT

public:
    ParameterDialog(QMainWindow *parent);
    int getDefaultHeight() override;
    int getDefaultWidth() override;
    const QString getSettingsKey() override;
    void show();

    QMainWindow *mainWindow;
    ParameterPanel *panel;

    const int defaultWidth = 400;
    const int defaultHeight = 320;
    const QString settingsKey = "ParameterDialog/size";

};

#endif // PARAMETERPANEL_H
