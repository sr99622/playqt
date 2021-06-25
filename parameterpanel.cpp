#include "parameterpanel.h"
#include "mainwindow.h"

ParameterDialog::ParameterDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Set Parameter");
    panel = new ParameterPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);
}


int ParameterDialog::getDefaultHeight()
{
    return 320;
}


void ParameterDialog::show()
{
    panel->setCmdLine();
    QDialog::show();
}

OptionBox::OptionBox(QWidget *parent) : QComboBox(parent)
{

}

void OptionBox::keyPressEvent(QKeyEvent *event)
{
    keyInput = true;
    QComboBox::keyPressEvent(event);
}

void OptionBox::mousePressEvent(QMouseEvent *event)
{
    keyInput = false;
    QComboBox::mousePressEvent(event);
}

StoredOption::StoredOption(const QString& text) : QListWidgetItem(text)
{

}

SavedCmdLines::SavedCmdLines(QMainWindow *parent) : QListWidget(parent)
{
    mainWindow = parent;
}

void SavedCmdLines::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Delete) {
        int row = currentRow();
        QListWidgetItem *item = takeItem(row);
        delete item;

        QString tag = "ParameterPanel/savedCmdLine_" + QString::number(row);
        if (MW->settings->contains(tag)) {
            cout << tag.toStdString() << endl;
            MW->settings->remove(tag);
        }
    }
    else if (event->key() == Qt::Key_Return) {
        QListWidgetItem *item = currentItem();
        MW->parameterDialog->panel->itemDoubleClicked(item);
    }
    QListWidget::keyPressEvent(event);
}

ParameterPanel::ParameterPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    options = new OptionBox(mainWindow);
    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (!(MW->co->options[i].flags & OPT_EXIT || MW->co->options[i].flags & OPT_NO_GUI))
            options->addItem(MW->co->options[i].help, QVariant(i));
    }
    parameter = new QLineEdit();
    QLabel *lbl00 = new QLabel("Command Line Equivalent: ");
    cmdLineEquiv = new QLabel();
    cmdLineEquiv->setStyleSheet("QLabel { background-color : #3E4754; color : #C6D9F2; }");

    QPushButton *set = new QPushButton("Set");
    QPushButton *clear = new QPushButton("Clear");
    clear->setFocusPolicy(Qt::NoFocus);

    QWidget *commandPanel = new QWidget(this);
    QGridLayout *cpLayout = new QGridLayout();
    QGroupBox *cpGroup = new QGroupBox("Set Command Line");
    cpLayout->addWidget(options,         0, 0, 1, 2);
    cpLayout->addWidget(parameter,       1, 0, 1, 2);
    cpLayout->addWidget(lbl00,           2, 0, 1, 2);
    cpLayout->addWidget(cmdLineEquiv,    3, 0, 1, 2);
    cpLayout->addWidget(clear,           4, 0, 1, 1, Qt::AlignCenter);
    cpLayout->addWidget(set,             4, 1, 1, 1, Qt::AlignCenter);
    cpLayout->setRowStretch(3, 10);
    cpGroup->setLayout(cpLayout);
    QHBoxLayout *gLayout = new QHBoxLayout();
    gLayout->addWidget(cpGroup);
    commandPanel->setLayout(gLayout);

    QLabel *lbl01 = new QLabel("Save the current command line to Profile: ");
    cmdLineName = new QLineEdit();
    QPushButton *saveCmdLine = new QPushButton("Save");
    QFontMetrics fm = saveCmdLine->fontMetrics();
    saveCmdLine->setMaximumWidth(fm.boundingRect("Save").width() * 1.6);
    savedCmdLines = new SavedCmdLines(mainWindow);
    QPushButton *clearSavedCmdLines = new QPushButton("Clear");

    QWidget *storagePanel = new QWidget(this);
    QGridLayout *spLayout = new QGridLayout();
    QGroupBox *spGroup = new QGroupBox("Save Command Line");
    spLayout->addWidget(lbl01,              0, 0, 1, 5, Qt::AlignCenter);
    spLayout->addWidget(cmdLineName,        1, 0, 1, 4);
    spLayout->addWidget(saveCmdLine,        1, 4, 1, 1);
    spLayout->addWidget(savedCmdLines,      2, 0, 1, 5);
    spLayout->addWidget(clearSavedCmdLines, 3, 2, 1, 1);
    spGroup->setLayout(spLayout);
    QHBoxLayout *sLayout = new QHBoxLayout();
    sLayout->addWidget(spGroup);
    storagePanel->setLayout(sLayout);

    QGridLayout *mainLayout = new QGridLayout();
    commandPanel->setMaximumWidth(400);
    mainLayout->addWidget(commandPanel,   0, 0, 1, 1);
    mainLayout->addWidget(storagePanel,   0, 1, 1, 1);

    setLayout(mainLayout);

    connect(set, SIGNAL(clicked()), this, SLOT(set()));
    connect(clear, SIGNAL(clicked()), this, SLOT(clear()));
    connect(options, SIGNAL(currentIndexChanged(int)), this, SLOT(optionChanged(int)));
    connect(parameter, SIGNAL(returnPressed()), this, SLOT(parameterEntered()));
    connect(saveCmdLine, SIGNAL(clicked()), this, SLOT(saveCmdLine()));
    connect(savedCmdLines, SIGNAL(itemDoubleClicked(QListWidgetItem*)), this, SLOT(itemDoubleClicked(QListWidgetItem*)));
    connect(clearSavedCmdLines, SIGNAL(clicked()), this, SLOT(clearSavedCmdLines()));
}

void ParameterPanel::clearSavedCmdLines()
{
    cout << "clearSavedCmdLines" << endl;
    bool reading = true;
    int i = 0;
    while (reading) {
        QString tag = "ParameterPanel/savedCmdLine_" + QString::number(i);
        if (MW->settings->contains(tag)) {
            cout << tag.toStdString() << endl;
            MW->settings->remove(tag);
            i++;
        }
        else {
            reading = false;
        }
    }

    savedCmdLines->clear();
}

const QString StoredOption::toString()
{
    QString str;
    QTextStream(&str) << "{" << text() << "}" << arg;
    return str;
}

void ParameterPanel::saveSettings(QSettings *settings)
{
    for (int i = 0; i < savedCmdLines->count(); i++) {
        QString tag = "ParameterPanel/savedCmdLine_" + QString::number(i);
        QString arg = ((StoredOption*)savedCmdLines->item(i))->toString();
        settings->setValue(tag, arg);
    }
}

void ParameterPanel::restoreSettings(QSettings *settings)
{
    bool reading = true;
    int i = 0;
    while (reading) {
        QString tag = "ParameterPanel/savedCmdLine_" + QString::number(i);
        if (settings->contains(tag)) {
            QString arg = settings->value(tag).toString();
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
            set(option_index, cmd_arg.toLatin1().data());
        }
        else {
            reading = false;
        }
    }
}

void ParameterPanel::saveCmdLine()
{
    cout << cmdLineEquiv->text().toStdString() << endl;
    StoredOption *storedOption = new StoredOption(cmdLineName->text());
    storedOption->arg = getOptionStorageString();
    savedCmdLines->addItem(storedOption);
    cmdLineName->setText("");
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
    options->setFocus();
}

void ParameterPanel::optionChanged(int index)
{
    parameter->setText("");
    if (!options->keyInput)
        parameter->setFocus();
}

void ParameterPanel::set()
{
    QString name = options->currentText();
    int index = MW->co->findOptionIndexByHelp(name);
    char *arg = parameter->text().toLatin1().data();
    set(index, arg);
}

void ParameterPanel::set(int option_index, char *arg)
{
    OptionDef *po = &MW->co->options[option_index];
    void *dst = po->u.dst_ptr;
    //char *arg = parameter->text().toLatin1().data();
    const char *opt = po->name;

    if (po->flags & OPT_STRING) {
        char *str;
        str = av_strdup(arg);
        av_freep(dst);
        if (!str) {
            cout << "ParameterPanel::set dst string error" << endl;
            return;
        }
        *(char **)dst = str;
    } else if (po->flags & OPT_BOOL || po->flags & OPT_INT) {
        *(int *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT_MIN, INT_MAX);
    } else if (po->flags & OPT_INT64) {
        *(int64_t *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT64_MIN, INT64_MAX);
    } else if (po->flags & OPT_TIME) {
        *(int64_t *)dst = parse_time_or_die(opt, arg, 1);
    } else if (po->flags & OPT_FLOAT) {
        *(float *)dst = parse_number_or_die(opt, arg, OPT_FLOAT, -INFINITY, INFINITY);
    } else if (po->flags & OPT_DOUBLE) {
        *(double *)dst = parse_number_or_die(opt, arg, OPT_DOUBLE, -INFINITY, INFINITY);
    } else if (po->u.func_arg) {
        int ret = po->u.func_arg(NULL, opt, arg);
        if (ret < 0) {
            cout << "ParameterPanel::set func_arg error" << endl;
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
    //MW->co->opt_add_vfilter(NULL, NULL, "");
    MW->co->video_codec_name = 0;
    MW->co->audio_disable = 0;
    MW->co->startup_volume = 100;

    saved_options.clear();

    setCmdLine();
}
