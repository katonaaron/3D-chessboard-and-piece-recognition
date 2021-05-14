#include "stdafx.h"
#include "Menu.h"

void Menu::show(std::istream& is, std::ostream& os) {
	size_t option;

	do
	{
		system("cls"); // TODO
		cv::destroyAllWindows(); // TODO

		os << "Menu:\n";

		int i = 1;
		for (const auto& option : options) {
			os << i++ << " - " << option.first << "\n";
		}

		os << "0 - Exit\n\n";
		os << "Option: ";
		os.flush();

		is >> option;
		is.ignore();

		if (option >= 1 && option <= options.size()) {
			options[option - 1].second();
		}
		else if (option != 0) {
			os << "Invalid option. Press any key..\n";
			is.get();
		}
	} while (option != 0);
}
