use crate::test::{TestAction, run_test_actions};

const TEST_HARNESS: &str = r#"
function assert(condition, message) {
    if (!condition) {
        if (!message) {
            message = "Assertion failed";
        }
        throw new Error(message);
    }
}

function assert_eq(a, b, message) {
    if (a !== b) {
        throw new Error(`${message} (${JSON.stringify(a)} !== ${JSON.stringify(b)})`);
    }
}
"#;

#[test]
fn url_basic() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com:8080/path/to/resource?query#fragment");
                assert(url instanceof URL);
                assert_eq(url.href, "https://example.com:8080/path/to/resource?query#fragment");
                assert_eq(url.protocol, "https:");
                assert_eq(url.host, "example.com:8080");
                assert_eq(url.hostname, "example.com");
                assert_eq(url.port, "8080");
                assert_eq(url.pathname, "/path/to/resource");
                assert_eq(url.search, "?query");
                assert_eq(url.hash, "#fragment");
            "##,
        ),
    ]);
}

#[test]
fn url_base() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com:8080/path/to/resource?query#fragment", "http://example.org/");
                assert_eq(url.href, "https://example.com:8080/path/to/resource?query#fragment");
                assert_eq(url.protocol, "https:");
                assert_eq(url.host, "example.com:8080");
                assert_eq(url.hostname, "example.com");
                assert_eq(url.port, "8080");
                assert_eq(url.pathname, "/path/to/resource");
                assert_eq(url.search, "?query");
                assert_eq(url.hash, "#fragment");
            "##,
        ),
        TestAction::run(
            r##"
                url = new URL("/path/to/resource?query#fragment", "http://example.org/");
                assert_eq(url.href, "http://example.org/path/to/resource?query#fragment");
                assert_eq(url.protocol, "http:");
                assert_eq(url.host, "example.org");
                assert_eq(url.hostname, "example.org");
                assert_eq(url.port, "");
                assert_eq(url.pathname, "/path/to/resource");
                assert_eq(url.search, "?query");
                assert_eq(url.hash, "#fragment");
            "##,
        ),
    ]);
}

#[test]
fn url_setters() {
    // These were double checked against Firefox.
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com:8080/path/to/resource?query#fragment");
                url.protocol = "http:";
                url.host = "example.org:80"; // Since protocol is http, port is removed.
                url.pathname = "/new/path";
                url.search = "?new-query";
                url.hash = "#new-fragment";
                assert_eq(url.href, "http://example.org/new/path?new-query#new-fragment");
                assert_eq(url.protocol, "http:");
                assert_eq(url.host, "example.org");
                assert_eq(url.hostname, "example.org");
                assert_eq(url.port, "");
                assert_eq(url.pathname, "/new/path");
                assert_eq(url.search, "?new-query");
                assert_eq(url.hash, "#new-fragment");
            "##,
        ),
    ]);
}

#[test]
fn url_static_methods() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                assert(URL.canParse("http://example.org/new/path?new-query#new-fragment"));
                assert(!URL.canParse("http//:example.org/new/path?new-query#new-fragment"));
                assert(!URL.canParse("http://example.org/new/path?new-query#new-fragment", "http:"));
                assert(URL.canParse("/new/path?new-query#new-fragment", "http://example.org/"));
            "##,
        ),
    ]);
}

#[test]
fn url_search_params_reads_a_query() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com/p?a=1&b=2&a=3");
                params = url.searchParams;
                assert(params instanceof URLSearchParams);
                assert_eq(params.get("a"), "1");
                assert_eq(params.get("b"), "2");
                assert_eq(params.get("missing"), null);
                assert_eq(params.getAll("a").join(","), "1,3");
                assert_eq(params.has("b"), true);
                assert_eq(params.has("b", "2"), true);
                assert_eq(params.has("b", "9"), false);
                assert_eq(params.size, 3);
                assert_eq(params.toString(), "a=1&b=2&a=3");
            "##,
        ),
    ]);
}

#[test]
fn url_search_params_can_be_walked() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com/p?a=1&b=2");

                // `forEach` is the one a router reaches for, and the argument
                // order is (value, name), not the other way round.
                seen = [];
                url.searchParams.forEach((value, name) => seen.push(name + "=" + value));
                assert_eq(seen.join("&"), "a=1&b=2");

                assert_eq([...url.searchParams.keys()].join(","), "a,b");
                assert_eq([...url.searchParams.values()].join(","), "1,2");
                assert_eq([...url.searchParams.entries()].map(p => p.join("=")).join("&"), "a=1&b=2");
                assert_eq([...url.searchParams].map(p => p.join("=")).join("&"), "a=1&b=2");
            "##,
        ),
    ]);
}

#[test]
fn url_search_params_is_a_view_of_its_url() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                url = new URL("https://example.com/p?a=1");

                // Writing through the params changes the URL.
                url.searchParams.append("b", "2");
                assert_eq(url.search, "?a=1&b=2");
                assert_eq(url.href, "https://example.com/p?a=1&b=2");

                // And writing the URL is visible through the params.
                url.search = "?c=3";
                assert_eq(url.searchParams.get("c"), "3");
                assert_eq(url.searchParams.get("a"), null);

                url.searchParams.set("c", "4");
                assert_eq(url.search, "?c=4");

                // Emptying the list removes the query rather than leaving a
                // bare question mark behind.
                url.searchParams.delete("c");
                assert_eq(url.search, "");
                assert_eq(url.href, "https://example.com/p");
            "##,
        ),
    ]);
}

#[test]
fn url_search_params_constructs_from_every_init() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                assert_eq(new URLSearchParams().toString(), "");
                assert_eq(new URLSearchParams("?a=1&b=2").toString(), "a=1&b=2");
                assert_eq(new URLSearchParams("a=1&b=2").toString(), "a=1&b=2");
                assert_eq(new URLSearchParams([["a", "1"], ["b", "2"]]).toString(), "a=1&b=2");
                assert_eq(new URLSearchParams({ a: "1", b: "2" }).toString(), "a=1&b=2");

                // A copy is independent: it must not stay a view over the URL
                // the original belonged to.
                url = new URL("https://example.com/p?a=1");
                copy = new URLSearchParams(url.searchParams);
                copy.set("a", "changed");
                assert_eq(url.search, "?a=1");
                assert_eq(copy.get("a"), "changed");
            "##,
        ),
    ]);
}

#[test]
fn url_search_params_encodes_and_sorts() {
    run_test_actions([
        TestAction::run(TEST_HARNESS),
        TestAction::run(
            r##"
                params = new URLSearchParams();
                params.append("q", "a b&c=d");
                assert_eq(params.toString(), "q=a+b%26c%3Dd");
                assert_eq(params.get("q"), "a b&c=d");

                sorted = new URLSearchParams("c=3&a=1&b=2");
                sorted.sort();
                assert_eq(sorted.toString(), "a=1&b=2&c=3");
            "##,
        ),
    ]);
}
