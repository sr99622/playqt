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

class FilterPanel : public QWidget
{
    Q_OBJECT

public:
    FilterPanel(QMainWindow *parent);
    ~FilterPanel() override;
    void addFilters();
    void idle();
    void saveSettings(QSettings *settings);
    void restoreSettings(QSettings *settings);
    bool isFilterActive(Filter *filter);
    bool isFilterActive(QString filter_name);
    Filter *getFilterByName(QString filter_name);
    Filter *getCurrentFilter();

    FilterListView *leftView;
    FilterListModel *leftModel;
    FilterListView *rightView;
    FilterListModel *rightModel;
    QPushButton *moveLeftButton;
    QPushButton *moveRightButton;
    QPushButton *moveUpButton;
    QPushButton *moveDownButton;

    QTabWidget *tabWidget;
    QWidget *bottomPanel;
    QMainWindow *mainWindow;
    QCheckBox *engageFilter;

    QVector<Filter*> filters;

public slots:
    void moveLeft();
    void moveRight();
    void moveUp();
    void moveDown();
    void initializeFilters();
    void panelShow(int index);
    void panelHide(int index);

};

class FilterDialog : public PanelDialog
{
    Q_OBJECT

public:
    FilterDialog(QMainWindow *parent);
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    int getDefaultWidth() override;
    int getDefaultHeight() override;
    const QString getSettingsKey() override;
    FilterPanel *panel;
    QMainWindow *mainWindow;

    const int defaultWidth = 520;
    const int defaultHeight = 720;
    const QString settingsKey = "FilterPanel/size";

};

#endif // FILTERPANEL_H
