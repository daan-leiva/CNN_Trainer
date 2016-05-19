#include <iostream>
#include <doublefann.h>
#include <fstream>
#include <string>
#include <limits>
#include <PathCch.h>

using namespace std;

const unsigned int num_input = 5400;
const unsigned int num_output = 1;
const unsigned int num_layers = 3;
const unsigned int num_neurons_hidden = 5400 / 2;
const float desired_error = (const float) 0.0001;
const unsigned int max_epochs = 100;
const unsigned int epochs_between_reports = 1;
string log_file_name = "TrainingLog.txt";
string current_epoch_file_name = "CurrentEpoch.txt";

int main()
{
	// fann pointer
	struct fann *ann;
	// structure shape 5400 -> 2700 -> 1
	const unsigned int layers[3] = { num_input, num_neurons_hidden, num_output };

	// get the number of data training files
	// that will be used to train Neural network
	int num_training_files = 0;
	cout << "Enter number of training files: ";
	cin >> num_training_files;
	while (cin.fail() || num_training_files < 1)
	{
		cout << "Invalid input. Positive non-zero integers only." << endl;
		cin.clear();
		cin.ignore(256, '\n');
		cout << "Enter number of training files: ";
		cin >> num_training_files;
	}
	
	// files are in the format name0, name1, name2
	// need to get the prefix "name" from user
	string file_prefix;
	string data_directory;
	cout << "Enter directory name containing all data files (do no include final '\'): ";
	getline(cin, data_directory);
	cout << "Enter prefix name of file: ";
	getline(cin, file_prefix);

	// check that all files exist
	for (int i = 0; i < num_training_files; i++) {
		ifstream f(data_directory + '/'  +file_prefix + to_string(i) + ".data");
		if (!f.good())
		{
			cout << "Data file list incomplete or cannot be accessed\n";
			cout << "Exiting program" << endl;
			return 0;
		}
	}
	// variable holds partial name of data file (just missing hte indexes + .data
	string partial_training_data_file_path = data_directory + '/' + file_prefix;

	// let user know that all files were found
	cout << "Files found. Access OK." << endl;

	// check if the user wants to store a new fann or use an old one
	// validate input
	bool create_new_fann = false;
	cout << "Create a new FANN? (no (0), yes (1))" << endl;
	cin >> create_new_fann;
	while (cin.fail())
	{
		cout << "Invalid input. 0 or 1 only." << endl;
		cin.clear();
		cin.ignore(256, '\n');
		cout << "Create a new FANN? (no (0), yes (1))" << endl;
		cin >> create_new_fann;
	}

	// get the name of the file
	string ann_file_name;
	string ann_file_directory;
	string ann_file_path;
	string log_file_path;
	string current_epoch_file_path;
	if (create_new_fann)
	{
		cout << "Enter name for new Neural Network file (no extension): ";
		getline(cin, ann_file_name);
		// gets path of source file
		char path_char[MAX_PATH];
		GetCurrentDirectoryA(MAX_PATH, path_char);
		ann_file_directory.assign(path_char);
		// create path name
		ann_file_path = ann_file_directory + "/" + ann_file_name + ".net";
		log_file_path = ann_file_directory + "/" + log_file_name;
		current_epoch_file_path = ann_file_directory + "/" + current_epoch_file_name;
	}
	else
	{
		cout << "Enter directory for saved Neural Network file (which should also include epoch/log file): ";
		getline(cin, ann_file_name);
		cout << "Enter name for saved Neural Network file (without '/'): ";
		getline(cin, ann_file_directory);
		// create path name
		ann_file_path = ann_file_directory + "/" + ann_file_name + ".net";
		log_file_path = ann_file_directory + "/" + log_file_name;
		current_epoch_file_path = ann_file_directory + "/" + current_epoch_file_name;
		// check if path exists
		ifstream ann_f(ann_file_path);
		ifstream log_f(log_file_path);
		ifstream epoch_f(current_epoch_file_path);
		while (!ann_f.good() || !log_f.good() || !epoch_f.good())
		{
			cout << "ANN file does not exist or cannot be accessed (or log/epoch file)\n";
			cout << "Enter directory for saved Neural Network file: ";
			getline(cin, ann_file_name);
			cout << "Enter name for saved Neural Network file (without '/'): ";
			getline(cin, ann_file_directory);
			// create path name
			ann_file_path = ann_file_directory + "/" + ann_file_name + ".net";
			log_file_path = ann_file_directory + "/" + log_file_name;
			current_epoch_file_path = ann_file_directory + "/" + current_epoch_file_name;
			ifstream ann_f(ann_file_path);
			ifstream log_f(log_file_path);
			ifstream epoch_f(current_epoch_file_path);
		}
	}

	//
	// load old fann or create new one
	//
	if (create_new_fann)
	{
		// create fann
		// start files from scratch
		ann = fann_create_standard_array(num_layers, layers);
		fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
		fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
		// reset log file
		ofstream log_file;
		log_file.open(log_file_path, ios::trunc);
		log_file.close();
		// reset epoch file
		ofstream epoch_file;
		epoch_file.open(current_epoch_file_path, ios::trunc);
		epoch_file << "Current epoch:" << endl;
		epoch_file << "0" << endl;
		epoch_file.close();
		// save ann file
		fann_save(ann, ann_file_path.c_str());
	}
	else
	{
		fann_create_from_file(ann_file_path.append(".net").c_str());
	}

	// get epoch num from epoch file
	ifstream current_epoch_file(current_epoch_file_path);
	string line;
	int epoch;
	getline(current_epoch_file, line);
	getline(current_epoch_file, line);
	epoch = stoi(line);
	current_epoch_file.close();

	// training data pointer
	struct fann_train_data *training_data;

	float MSE = 0.0f;
	int bitfai_count = 0;
	// training loop
	// open log file
	ofstream log;
	
	for (int epoch_num = 0; epoch_num < max_epochs; epoch_num) {
		// open log
		log.open(log_file_path, ios::app);
		MSE = 0;
		bitfai_count = 0;
		for (int i = 0; i < num_training_files; i++)
		{
			// get current file path
			string training_data_file_path = partial_training_data_file_path + to_string(i) + ".data";
			// create training data
			training_data = fann_read_train_from_file(training_data_file_path.c_str());
			// run epoch on training data
			MSE += fann_train_epoch(ann, training_data);
		}
		bitfai_count = fann_get_bit_fail(ann);
		epoch++;
		// write to log
		log << "Epoch: " << epoch << endl;
		log << "MSE: " << MSE << endl;
		log << "Bit Fail: " << fann_get_bit_fail(ann) << endl;
		log.close();
		// write to screen
		cout << "Epoch: " << epoch << endl;
		cout << "MSE: " << MSE << endl;
		cout << "Bit Fail: " << fann_get_bit_fail(ann) << endl;
	}
	
	// write final epoch number
	ofstream current_epoch_write;
	current_epoch_write.open(current_epoch_file_path, ios::trunc);
	current_epoch_write << "Current Epoch" << endl << epoch;
	// close epoch file
	current_epoch_write.close();

	// save and destroy FANN
	fann_save(ann, "CNN_Edge_Detection_FANN_test.net");
	fann_destroy(ann);

	return 0;
}