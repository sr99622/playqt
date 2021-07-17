/*******************************************************************************
* filterpanel.h
*
* Copyright (c) 2020 Stephen Rhodes
*
* This program is free software; you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation; either version 2 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along
* with this program; if not, write to the Free Software Foundation, Inc.,
* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*******************************************************************************/

#ifndef FILTERPANEL_H
#define FILTERPANEL_H

#include "filterlistview.h"
#include "filterlistmodel.h"
#include "Utilities/paneldialog.h"
#include <QWidget>
#include <QVector>
#include <QPushButton>
#include <QMainWindow>
#include <QCheckBox>
#include <QLabel>

class FilterPanel : public QWidget
{
    Q_OBJECT

public:
    FilterPanel(QMainWindow *parent);
    ~FilterPanel() override;
    void idle();
    void saveSettings(QSettings *settings);
    void restoreSettings(QSettings *settings);
    bool isFilterActive(Filter *filter);
    bool isFilterActive(QString filter_name);
    Filter *getFilterByName(QString filter_name);
    Filter *getCurrentFilter();

    QMainWindow *mainWindow;
    FilterListView *leftView;
    FilterListModel *leftModel;
    FilterListView *rightView;
    FilterListModel *rightModel;
    QPushButton *moveLeftButton;
    QPushButton *moveRightButton;
    QPushButton *moveUpButton;
    QPushButton *moveDownButton;
    QTabWidget *tabWidget;
    //QWidget *bottomPanel;
    QCheckBox *engageFilter;
    QLabel *filterTime;

    QVector<Filter*> filters;

public slots:
    void moveLeft();
    void moveRight();
    void moveUp();
    void moveDown();
    void initializeFilters();
    void engage(int);
    void panelShow(int);
    void tabChanged(int);
    //void panelHide(int index);

};

class FilterDialog : public PanelDialog
{
    Q_OBJECT

public:
    FilterDialog(QMainWindow *parent);
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void closeEvent(QCloseEvent *event) override;
    int getDefaultWidth() override;
    int getDefaultHeight() override;
    QString getSettingsKey() const override;
    FilterPanel *panel;
    QMainWindow *mainWindow;

    const int defaultWidth = 520;
    const int defaultHeight = 600;
    const QString settingsKey = "FilterPanel/geometry";

};

#endif // FILTERPANEL_H
