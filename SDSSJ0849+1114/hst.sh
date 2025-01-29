curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=IC5O01020%2Fic5o01020_asn.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/IC5O01020/ic5o01020_asn.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=IC5O01020%2Fic5o01020_drz.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/IC5O01020/ic5o01020_drz.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=IC5O01020%2Fic5o01edq_flt.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/IC5O01020/ic5o01edq_flt.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=IC5O01020%2Fic5o01eeq_flt.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/IC5O01020/ic5o01eeq_flt.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=IC5O01020%2Fic5o01efq_flt.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/IC5O01020/ic5o01efq_flt.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9w010_asn.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9w010_asn.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9w010_drc.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9w010_drc.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9w010_drz.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9w010_drz.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9wcxq_flt.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9wcxq_flt.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9wcxq_flc.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9wcxq_flc.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9wcyq_flt.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9wcyq_flt.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=JDRW9W010%2Fjdrw9wcyq_flc.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/JDRW9W010/jdrw9wcyq_flc.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=U67L7001R%2Fu67l7001r_drz.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/U67L7001R/u67l7001r_drz.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=U67L7001R%2Fu67l7001r_c0f.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/U67L7001R/u67l7001r_c0f.fits" --fail --create-dirs

curl -L -X GET "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product?product_name=U67L7001R%2Fu67l7001r_c1f.fits" --output "MAST_2024-10-01T05_54_16.720Z/HST/U67L7001R/u67l7001r_c1f.fits" --fail --create-dirs

curl -H "Authorization: token $MAST_API_TOKEN" -L -X POST "https://mast.stsci.edu/search/hst/api/v0.1/retrieve_product_zip" -F "mission=HST" -F "structure=nested" -F uri="IC5O01020/ic5o01020_asn.fits,IC5O01020/ic5o01020_drz.fits,IC5O01020/ic5o01edq_flt.fits,IC5O01020/ic5o01eeq_flt.fits,IC5O01020/ic5o01efq_flt.fits,JDRW9W010/jdrw9w010_asn.fits,JDRW9W010/jdrw9w010_drc.fits,JDRW9W010/jdrw9w010_drz.fits,JDRW9W010/jdrw9wcxq_flt.fits,JDRW9W010/jdrw9wcxq_flc.fits,JDRW9W010/jdrw9wcyq_flt.fits,JDRW9W010/jdrw9wcyq_flc.fits,U67L7001R/u67l7001r_drz.fits,U67L7001R/u67l7001r_c0f.fits,U67L7001R/u67l7001r_c1f.fits" -F "manifestonly=true" --output "MAST_2024-10-01T05_54_16.720Z/MANIFEST.html" --fail --create-dirs


