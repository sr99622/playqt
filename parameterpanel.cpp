#include "parameterpanel.h"
#include "mainwindow.h"

ParameterPanel::ParameterPanel(QMainWindow *parent) : Panel(parent)
{
    mainWindow = parent;

    options = new QComboBox();
    options->setPlaceholderText("-");
    //connect(options, SIGNAL(activated(int)), this, SLOT(comboActivated(int)));

    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (!(MW->co->options[i].flags & OPT_EXIT || MW->co->options[i].flags & OPT_NO_GUI))
            options->addItem(MW->co->options[i].help, QVariant(i));
    }
    connect(options, SIGNAL(currentIndexChanged(int)), this, SLOT(optionChanged(int)));

    parameter = new QLineEdit();
    connect(parameter, SIGNAL(returnPressed()), this, SLOT(parameterEntered()));

    QLabel *lbl00 = new QLabel("Command Line Equivalent: ");
    cmdLineEquiv = new QLabel();
    cmdLineEquiv->setStyleSheet("QLabel { background-color : #3E4754; color : #C6D9F2; }");
    cmdLineEquiv->setWordWrap(true);

    QPushButton *set = new QPushButton("Set");
    QPushButton *clear = new QPushButton("Clear");
    int width = clear->fontMetrics().boundingRect("WWWWW").width();
    set->setMinimumWidth(width);
    clear->setMinimumWidth(width);
    set->setFocusPolicy(Qt::NoFocus);
    clear->setFocusPolicy(Qt::NoFocus);
    connect(set, SIGNAL(clicked()), this, SLOT(set()));
    connect(clear, SIGNAL(clicked()), this, SLOT(clear()));

    QWidget *commandPanel = new QWidget(this);
    QGridLayout *cpLayout = new QGridLayout();
    QGroupBox *cpGroup = new QGroupBox("Set Command Line");
    cpLayout->addWidget(options,         0, 0, 1, 3);
    cpLayout->addWidget(parameter,       1, 0, 1, 3);
    cpLayout->addWidget(lbl00,           2, 0, 1, 3);
    cpLayout->addWidget(cmdLineEquiv,    3, 0, 1, 3);
    cpLayout->addWidget(clear,           4, 0, 1, 1, Qt::AlignCenter);
    cpLayout->addWidget(set,             4, 2, 1, 1, Qt::AlignCenter);
    cpLayout->setRowStretch(3, 10);
    cpGroup->setLayout(cpLayout);
    QHBoxLayout *gLayout = new QHBoxLayout();
    gLayout->addWidget(cpGroup);
    commandPanel->setLayout(gLayout);

    QLabel *lbl01 = new QLabel("Save the current command: ");
    cmdLineName = new QLineEdit();
    connect(cmdLineName, SIGNAL(returnPressed()), this, SLOT(saveCmdLine()));
    QPushButton *saveCmdLine = new QPushButton("Save");
    QFontMetrics fm = saveCmdLine->fontMetrics();
    saveCmdLine->setMaximumWidth(fm.boundingRect("Save").width() * 1.6);
    saveCmdLine->setFocusPolicy(Qt::NoFocus);
    connect(saveCmdLine, SIGNAL(clicked()), this, SLOT(saveCmdLine()));

    savedCmdLines = new SavedCmdLines(mainWindow);
    restoreSettings();
    savedCmdLines->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(savedCmdLines, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(showContextMenu(const QPoint&)));
    //connect(savedCmdLines, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(itemDoubleClicked(QListWidgetItem*)));
    connect(savedCmdLines, SIGNAL(itemChanged(QListWidgetItem*)), this, SLOT(itemChanged(QListWidgetItem*)));

    QWidget *storagePanel = new QWidget(this);
    QGridLayout *spLayout = new QGridLayout();
    QGroupBox *spGroup = new QGroupBox("Save Command Line");
    spLayout->addWidget(lbl01,              0, 0, 1, 5, Qt::AlignCenter);
    spLayout->addWidget(cmdLineName,        1, 0, 1, 4);
    spLayout->addWidget(saveCmdLine,        1, 4, 1, 1);
    spLayout->addWidget(savedCmdLines,      2, 0, 1, 5);
    spGroup->setLayout(spLayout);
    QHBoxLayout *sLayout = new QHBoxLayout();
    sLayout->addWidget(spGroup);
    storagePanel->setLayout(sLayout);

    QGridLayout *mainLayout = new QGridLayout();
    mainLayout->addWidget(commandPanel,      0, 0, 1, 1);
    mainLayout->addWidget(storagePanel,      0, 1, 1, 1);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    setLayout(mainLayout);

    menu = new QMenu("SavedCmdLines Menu", savedCmdLines);
    QAction *remove = new QAction("Delete", this);
    QAction *rename = new QAction("Rename", this);
    connect(remove, SIGNAL(triggered()), savedCmdLines, SLOT(remove()));
    connect(rename, SIGNAL(triggered()), savedCmdLines, SLOT(rename()));
    menu->addAction(remove);
    menu->addAction(rename);

    savedCmdLines->setFocus();
    filters = ((OptionPanel*)MW->optionDialog->panel)->show_filters();
}

void ParameterPanel::apply()
{
    MW->runLoop();
}

void ParameterPanel::comboActivated(int index)
{
    cout << "ParameterPanel::comboActivated" << endl;
}

void ParameterPanel::itemChanged(QListWidgetItem *item)
{
    QString tag = cmdKey + QString::number(savedCmdLines->currentRow());
    QString arg = ((StoredOption*)item)->toString();
    MW->settings->setValue(tag, arg);
}

void ParameterPanel::showContextMenu(const QPoint &pos)
{
    QModelIndex index = savedCmdLines->indexAt(pos);
    if (index.isValid())
        menu->exec(savedCmdLines->mapToGlobal(pos));
}

void ParameterPanel::clearSettings()
{
    for (int i = 0; i < savedCmdLines->count(); i++) {
        QString tag = cmdKey + QString::number(i);
        MW->settings->remove(tag);
    }
}

void ParameterPanel::saveSettings()
{
    for (int i = 0; i < savedCmdLines->count(); i++) {
        QString tag = cmdKey + QString::number(i);
        QString arg = ((StoredOption*)savedCmdLines->item(i))->toString();
        MW->settings->setValue(tag, arg);
    }
}

void ParameterPanel::restoreSettings()
{
    bool reading = true;
    int i = 0;
    while (reading) {
        QString tag = cmdKey + QString::number(i);
        if (MW->settings->contains(tag)) {
            QString arg = MW->settings->value(tag).toString();
            int start = arg.indexOf("{") + 1;
            int stop = arg.indexOf("}");
            QString title = arg.sliced(start, stop - start);

            start = stop + 1;
            QString option_arg = arg.sliced(start, arg.length() - start);

            StoredOption *option = new StoredOption(title);
            option->arg = option_arg;
            savedCmdLines->addItem(option);

            i++;
        }
        else {
            reading = false;
        }
    }
}

void ParameterPanel::itemDoubleClicked(QListWidgetItem *item)
{
    QString arg = ((StoredOption*)item)->arg;
    cout << "ParameterPanel::itemDoubleClicked arg: " << arg.toStdString() << endl;
    bool reading = true;
    int stop = -1;
    while (reading) {
        int start = stop + 1;
        stop = arg.indexOf(";", start);
        if (stop > -1) {
            QString str_option = arg.sliced(start, stop - start);
            int delim = str_option.indexOf(",");

            QString cmd_name = str_option.sliced(0, delim);
            int option_index = MW->co->findOptionIndexByName(cmd_name);

            QString cmd_arg = str_option.sliced(delim +1, str_option.length() - delim);

            if (cmd_name == "ss" || cmd_name == "t") {
                bool ok;
                double raw_value = QString(cmd_arg.toLatin1().data()).toDouble(&ok);
                if (!ok) {
                    cout << "ParameterPanel::itemDoubleClicked number parse failure: "
                         << cmd_name.toStdString() << " " << cmd_arg.toStdString() << endl;
                    return;
                }
                int cmd_value = raw_value / 1000000;
                cmd_arg = QString::number(cmd_value);

            }
            set(option_index, cmd_arg);
        }
        else {
            reading = false;
        }
    }
}

void ParameterPanel::saveCmdLine()
{
    cout << "ParameterPanel::saveCmdLine" << cmdLineEquiv->text().toStdString() << endl;

    if (cmdLineEquiv->text().length() == 0) {
        QMessageBox::warning(this, "PlayQt", "No command line is currently set");
        return;
    }

    if (cmdLineName->text().length() == 0) {
        QMessageBox::warning(this, "PlayQt", "Command name is required");
        return;
    }

    StoredOption *storedOption = new StoredOption(cmdLineName->text());
    storedOption->arg = getOptionStorageString();
    savedCmdLines->addItem(storedOption);
    cmdLineName->setText("");
    saveSettings();
}

const QString ParameterPanel::getOptionStorageString()
{
    QString arg;
    for (int i = 0; i < saved_options.size(); i++) {
        cout << "number of saved options: " << saved_options.size() << endl;
        OptionDef option = saved_options[i];
        QTextStream(&arg) << option.name;
        if (option.flags & OPT_STRING) {
            char *str = *(char **) option.u.dst_ptr;
            QTextStream(&arg) << "," << str << ";";
        }
        else if (option.flags & OPT_BOOL) {
            QTextStream(&arg) << ",1;";
        }
        else if (option.flags & OPT_INT) {
            QTextStream(&arg) << "," << *(int *) option.u.dst_ptr << ";";
        }
        else if (!strcmp(option.name, "ss")) {
            QTextStream(&arg) << "," << MW->co->start_time << ";";
        }
        else if (!strcmp(option.name, "t")) {
            QTextStream(&arg) << "," << MW->co->duration << ";";
        }
        else if (!strcmp(option.name, "vf")) {
            bool first_pass = true;
            QTextStream(&arg) << ",";
            for (int j = 0; j < MW->co->nb_vfilters; j++) {
                if (!first_pass) {
                    QTextStream(&arg) << " ";
                }
                QTextStream(&arg) << MW->co->vfilters_list[j];
                first_pass = false;
            }
            QTextStream(&arg) << ";";
        }
        else if (!strcmp(option.name, "sync")) {
            QTextStream(&arg) << "," << MW->co->clock_sync << ";";
        }
    }
    return arg;
}

void ParameterPanel::setCmdLine()
{
    QString str;

    for (int i = 0; i < NUM_OPTIONS; i++) {
        OptionDef po = MW->co->options[i];
        if (po.flags & OPT_EXIT || po.flags & OPT_NO_GUI)
            continue;

        if (po.flags & OPT_STRING) {
            if (po.u.dst_ptr) {
                char *arg = *(char **) po.u.dst_ptr;
                if (arg)
                    QTextStream(&str) << " -" << po.name << " " << arg;
            }
        }
        else if (po.flags & OPT_BOOL) {
            if (po.u.dst_ptr) {
                int arg = *(int *) po.u.dst_ptr;
                if (!strcmp(po.name, "framedrop") || !strcmp(po.name, "infbuf")) {
                    if (arg > 0)
                        QTextStream(&str) << " -" << po.name;
                }
                else {
                    if (arg)
                        QTextStream(&str) << " -" << po.name;
                }
            }
        }
        else if (po.flags & OPT_INT) {
            if (po.u.dst_ptr) {
                int arg = *(int *) po.u.dst_ptr;
                if (!strcmp(po.name, "drp")) {
                    if (arg != -1) {
                        QTextStream(&str) << " -" << po.name << " " << arg;
                    }
                }
                else if (!strcmp(po.name, "volume")) {
                    if (arg < 100)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
                else if (!strcmp(po.name, "bytes")) {
                    if (arg > 0)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
                else {
                    if (arg)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
            }
        }
        else if (po.flags & OPT_INT64) {
            if (po.u.dst_ptr) {
                int64_t arg = *(int64_t *) po.u.dst_ptr;
                if (arg) {
                    QTextStream(&str) << " -" << po.name << " " << arg;
                }
            }
        }
        else if (po.flags & OPT_FLOAT) {
            if (po.u.dst_ptr) {
                int64_t arg = *(float *) po.u.dst_ptr;
                if (!strcmp(po.name, "seek_interval")) {
                    if (arg != 10) {
                        QTextStream(&str) << " -" << po.name << " " << arg;
                    }
                }
                else {
                    if (arg)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
            }
        }
        else {
            if (po.u.func_arg) {
                if (!strcmp(po.name, "ss")) {
                    int arg = MW->co->start_time;
                    if (arg)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
                if (!strcmp(po.name, "t")) {
                    int arg = MW->co->duration;
                    if (arg)
                        QTextStream(&str) << " -" << po.name << " " << arg;
                }
                if (!strcmp(po.name, "f")) {
                    const char *arg = MW->co->forced_format;
                    if (arg) {
                        QTextStream(&str) << " -" << po.name << " " << arg;
                    }
                }
                if (!strcmp(po.name, "sync")) {
                    const char *arg = MW->co->clock_sync;
                    if (arg) {
                        QTextStream(&str) << " -" << po.name << " " << arg;
                    }

                }
                if (!strcmp(po.name, "vf")) {
                    int arg = MW->co->nb_vfilters;
                    if (arg) {
                        QTextStream(&str) << " -" << po.name;
                        for (int i = 0; i < arg; i++) {
                            QTextStream(&str) << " " << MW->co->vfilters_list[i];
                        }
                        cout << endl;
                    }
                }
            }
        }
    }

    cmdLineEquiv->setText(str);
}

void ParameterPanel::parameterEntered()
{
    set();
    cmdLineName->setFocus();
}

void ParameterPanel::optionChanged(int index)
{
    parameter->setText("");
    //if (!options->keyInput)
    //    parameter->setFocus();
}

void ParameterPanel::set()
{
    QString name = options->currentText();
    int index = MW->co->findOptionIndexByHelp(name);
    //char *arg = parameter->text().toLatin1().data();
    set(index, parameter->text());
}

void ParameterPanel::set(int option_index, QString option_arg)
{
    cout << "option_index: " << option_index << endl;
    cout << "description: " << options->currentText().toStdString() << endl;

    if (option_index < 0) {
        QMessageBox::warning(this, "PlayQt", "No option specified, please select an option from the drop down box");
        return;
    }

    if (option_arg.length() == 0) {
        QMessageBox::warning(this, "PlayQt", "Incomplete parameter specification.  Arguments are required for all parameters");
        return;
    }
    cout << "option_arg: " << option_arg.toStdString() << endl;

    const QString str(option_arg);
    char *arg = str.toLatin1().data();

    cout << "str: " << str.toStdString() << endl;

    OptionDef *po = &MW->co->options[option_index];
    void *dst = po->u.dst_ptr;
    //char *arg = parameter->text().toLatin1().data();
    const char *opt = po->name;
    QString name(opt);
    cout << "opt: " << name.toStdString() << endl;
    bool ok;

    cout << "test 2" << endl;

    if (po->flags & OPT_STRING) {
        cout << "OPT_STRING" << endl;

        av_freep(dst);
        *(char **)dst = av_strdup(str.toLatin1().data());


        /*
        char *tmp;
        tmp = av_strdup(arg);
        av_freep(dst);

        if (!tmp) {
            cout << "ParameterPanel::set dst string error" << endl;
            return;
        }
        *(char **)dst = tmp;
        */
    }
    else if (po->flags & OPT_BOOL || po->flags & OPT_INT) {
        cout << "OPT_BOOL" << endl;
        QString(arg).toInt(&ok);
        if (!ok) {
            QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(arg));
            return;
        }
        *(int *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT_MIN, INT_MAX);
    }
    else if (po->flags & OPT_INT64) {
        cout << "OPT_INT64" << endl;
        QString(arg).toLong(&ok);
        if (!ok) {
            QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(arg));
            return;
        }
        *(int64_t *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT64_MIN, INT64_MAX);
    }
    else if (po->flags & OPT_TIME) {
        cout << "OPT_TIME" << endl;
        QString(arg).toLong(&ok);
        if (!ok) {
            QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(arg));
            return;
        }
        *(int64_t *)dst = parse_time_or_die(opt, arg, 1);
    }
    else if (po->flags & OPT_FLOAT) {
        cout << "OPT_FLOAT" << endl;
        QString(arg).toFloat(&ok);
        if (!ok) {
            QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(arg));
            return;
        }
        *(float *)dst = parse_number_or_die(opt, arg, OPT_FLOAT, -INFINITY, INFINITY);
    }
    else if (po->flags & OPT_DOUBLE) {
        cout << "OPT_DOUBLE" << endl;
        QString(arg).toDouble(&ok);
        if (!ok) {
            QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(arg));
            return;
        }
        *(double *)dst = parse_number_or_die(opt, arg, OPT_DOUBLE, -INFINITY, INFINITY);
    }

    else if (po->u.func_arg) {
        cout << "FUNC ARG: " << endl;
        cout << "str: " << str.toStdString() << endl;
        if (name == "ss" || name == "t") {
            str.toFloat(&ok);
            if (!ok) {
                QMessageBox::warning(this, "PlayQt", QString("Incorrect format %1 is not a number").arg(str));
                return;
            }
        }
        else if (name == "vf") {
            cout << "VIDEO FILTER" << endl;
            cout << "str: " << str.toStdString() << endl;
            QString base;
            int index = str.indexOf("=");
            if (index > -1)
                base = str.mid(0, index);
            else
                base = str;
            cout << "base: ----" << base.toStdString() << "---" << endl;

            if (!filters.contains(base.toLatin1().data())) {
                cout << filters.toStdString() << endl;
                QMessageBox::warning(this, "PlayQT", QString("Filter %1 not found").arg(str));
                return;
            }
        }

        cout << "arg: " << arg << endl;

        int ret = po->u.func_arg(NULL, opt, str.toLatin1().data());
        if (ret < 0) {
            QMessageBox::warning(this, "PlayQt", "Data entered in incorrect format");
            return;
        }
    }

    for (size_t i = 0; i < saved_options.size(); i++) {
        if (!strcmp(po->name, saved_options[i].name)) {
            saved_options.erase(saved_options.begin() + i);
            break;
        }
    }
    saved_options.push_back(*po);

    setCmdLine();
}

void ParameterPanel::addOptionToSaver(OptionDef option)
{
    for (size_t i = 0; i < saved_options.size(); i++) {
        if (!strcmp(option.name, saved_options[i].name)) {
            saved_options.erase(saved_options.begin() + i);
            break;
        }
    }
    saved_options.push_back(option);
}

void ParameterPanel::clear()
{
    parameter->setText("");
    MW->co->duration = AV_NOPTS_VALUE;
    MW->co->start_time = AV_NOPTS_VALUE;
    MW->co->video_codec_name = nullptr;
    MW->co->audio_codec_name = nullptr;
    MW->co->subtitle_codec_name = nullptr;
    MW->co->video_disable = 0;
    MW->co->audio_disable = 0;
    MW->co->subtitle_disable = 0;
    MW->co->startup_volume = 100;
    MW->co->wanted_stream_spec[AVMEDIA_TYPE_AUDIO] = 0;
    MW->co->wanted_stream_spec[AVMEDIA_TYPE_VIDEO] = 0;
    MW->co->wanted_stream_spec[AVMEDIA_TYPE_SUBTITLE] = 0;
    MW->co->clock_sync = nullptr;
    MW->co->av_sync_type = AV_SYNC_AUDIO_MASTER;
    MW->co->seek_by_bytes = -1;
    MW->co->seek_interval = 10;
    MW->co->fast = 0;
    MW->co->genpts = 0;
    MW->co->framedrop = -1;
    MW->co->lowres = 0;
    MW->co->decoder_reorder_pts = -1;

#if CONFIG_AVFILTER
    MW->co->opt_add_vfilter(NULL, NULL, "");
#endif

    saved_options.clear();

    setCmdLine();
}

ParameterDialog::ParameterDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Set Parameter");
    panel = new ParameterPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);

    defaultWidth = 400;
    defaultHeight = 320;
    settingsKey = "ParameterDialog/size";
}

void ParameterDialog::show()
{
    ((ParameterPanel*)panel)->setCmdLine();
    PanelDialog::show();
}

StoredOption::StoredOption(const QString& text) : QListWidgetItem(text)
{

}

const QString StoredOption::toString()
{
    QString str;
    QTextStream(&str) << "{" << text() << "}" << arg;
    return str;
}

SavedCmdLines::SavedCmdLines(QMainWindow *parent) : QListWidget(parent)
{
    mainWindow = parent;
}

void SavedCmdLines::remove()
{
    QString msg = QString("You are about to delete the command '%1'\nAre you sure you want to continue?").arg(currentItem()->text());
    QMessageBox::StandardButton result = QMessageBox::question(this, "PlayQt", msg);
    if (result == QMessageBox::Yes) {
        ((ParameterPanel*)MW->parameterDialog->panel)->clearSettings();

        int row = currentRow();
        QListWidgetItem *item = takeItem(row);
        delete item;

        /*
        QString tag = MW->parameterDialog->panel->cmdKey + QString::number(row);
        if (MW->settings->contains(tag)) {
            cout << tag.toStdString() << endl;
            MW->settings->remove(tag);
        }
        */

        ((ParameterPanel*)MW->parameterDialog->panel)->saveSettings();

    }
}

void SavedCmdLines::rename()
{
    QListWidgetItem *current_item = currentItem();
    current_item->setFlags(current_item->flags() | Qt::ItemIsEditable);
    editItem(current_item);
}

void SavedCmdLines::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Delete:
        remove();
        break;
    case Qt::Key_F2:
        rename();
        break;
    case Qt::Key_Return:
        if (currentItem())
            ((ParameterPanel*)MW->parameterDialog->panel)->itemDoubleClicked(currentItem());
        break;
    case Qt::Key_Escape:
        ((ParameterPanel*)MW->parameterDialog->panel)->clear();
        break;
    default:
        QListWidget::keyPressEvent(event);
    }

}

void SavedCmdLines::mouseDoubleClickEvent(QMouseEvent *event)
{
    QListWidgetItem *item = itemAt(event->pos());
    if (item)
        ((ParameterPanel*)MW->parameterDialog->panel)->itemDoubleClicked(item);
}
