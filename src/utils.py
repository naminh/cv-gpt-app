import os

from pypdf import PdfReader, PdfWriter


def decrypt_pdf(input_path, password):
    output_path = input_path.split(".")[0] + "_decrypted.pdf"
    with open(input_path, "rb") as input_file, open(output_path, "wb") as output_file:
        reader = PdfReader(input_file)
        reader.decrypt(password)

        writer = PdfWriter()

        for i in range(len(reader.pages)):
            writer.add_page(reader.pages[i])

        writer.write(output_file)

    return output_path


def decrypt_cv(input_path):
    p = os.getenv("PDF_PASS")
    return decrypt_pdf(input_path, p)
