#![allow(missing_docs, rustdoc::missing_crate_level_docs)]

use std::path::Path;
use std::{error::Error, fs::File};

use icu_provider_export::blob_exporter::BlobExporter;
use icu_provider_export::prelude::*;
use icu_provider_source::{CoverageLevel, SourceDataProvider};

/// Path to the directory where the exported data lives.
const EXPORT_PATH: &str = "core/icu_provider/data";

/// Brotli encoder settings for the exported blobs.
///
/// Quality 11 is the maximum and the window is the standard 2^22, which every
/// brotli decoder accepts without the large-window extension. Compression runs
/// once, offline, so the slow setting costs nothing at build or run time.
const BROTLI_BUFFER_SIZE: usize = 4096;
const BROTLI_QUALITY: u32 = 11;
const BROTLI_WINDOW: u32 = 22;

/// List of services used by `Intl` components.
///
/// This must be kept in sync with the list of implemented services for `Intl`.
const SERVICES: &[(&str, &[DataMarkerInfo])] = &[
    ("icu_casemap", icu_casemap::provider::MARKERS),
    ("icu_collator", icu_collator::provider::MARKERS),
    ("icu_datetime", icu_datetime::provider::MARKERS),
    ("icu_time", icu_time::provider::MARKERS),
    ("icu_decimal", icu_decimal::provider::MARKERS),
    ("icu_list", icu_list::provider::MARKERS),
    ("icu_locale", icu_locale::provider::MARKERS),
    ("icu_normalizer", icu_normalizer::provider::MARKERS),
    ("icu_plurals", icu_plurals::provider::MARKERS),
    ("icu_segmenter", icu_segmenter::provider::MARKERS),
];

fn export_for_service(
    service: &str,
    markers: &[DataMarkerInfo],
    provider: &SourceDataProvider,
    driver: ExportDriver,
) -> Result<(), Box<dyn Error>> {
    log::info!("Generating ICU4X data for service `{service}` with markers: {markers:#?}");

    let export_path = Path::new(EXPORT_PATH);
    let export_file = export_path.join(format!("{service}.postcard.br"));

    // The blob is written through a brotli encoder rather than stored raw. The
    // postcard data is highly compressible and `boa_icu_provider` embeds every
    // service in the binary, so shipping it compressed is the difference
    // between a 8.7MB and a 3.2MB `Intl`. The provider decompresses each
    // service lazily, on its first data request.
    let sink = brotli::CompressorWriter::new(
        File::create(export_file)?,
        BROTLI_BUFFER_SIZE,
        BROTLI_QUALITY,
        BROTLI_WINDOW,
    );

    driver
        .with_markers(markers.iter().copied())
        .export(provider, BlobExporter::new_with_sink(Box::new(sink)))?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    simple_logger::SimpleLogger::new()
        .env()
        .with_level(log::LevelFilter::Info)
        .init()?;

    // Removal will throw an error if the directory doesn't exist, hence
    // why we can ignore the error.
    let _unused = std::fs::remove_dir_all(EXPORT_PATH);
    std::fs::create_dir_all(EXPORT_PATH)?;

    let provider = &SourceDataProvider::new();
    let locales = provider
        .locales_for_coverage_levels([CoverageLevel::Modern])?
        .into_iter()
        .map(DataLocaleFamily::with_descendants)
        .chain([
            // test262 assumes the en-US locale does not fallback.
            // Required by https://github.com/tc39/test262/blob/a073f479f80b336256b7fc4e04700c827293e2fe/test/intl402/ListFormat/prototype/resolvedOptions/type.js
            DataLocaleFamily::single(locale!("en-US").into()),
            // test262 uses the Manx locale.
            // Required by https://github.com/tc39/test262/blob/a073f479f80b336256b7fc4e04700c827293e2fe/test/intl402/PluralRules/prototype/resolvedOptions/plural-categories-order.js
            DataLocaleFamily::with_descendants(locale!("gv").into()),
        ]);

    let driver = ExportDriver::new(
        locales,
        DeduplicationStrategy::None.into(),
        LocaleFallbacker::try_new_unstable(provider)?,
    )
    .with_additional_collations([String::from("search*")])
    .with_recommended_segmenter_models();

    for (service, keys) in SERVICES {
        export_for_service(service, keys, provider, driver.clone())?;
    }

    Ok(())
}
