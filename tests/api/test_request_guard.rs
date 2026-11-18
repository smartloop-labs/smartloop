//! The request guard in front of every route: a request must name this
//! machine as its `Host`, and a browser's must come from a page of this
//! machine's, or be the platform's sign-in hand-off.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use tower::ServiceExt;

/// Send `method uri` through the full app with `headers`; its status and
/// `access-control-allow-origin`.
async fn send(method: Method, uri: &str, headers: &[(&str, &str)]) -> (StatusCode, Option<String>) {
    let mut builder = Request::builder().method(method).uri(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    let response = smartloop::api::app().oneshot(builder.body(Body::empty()).unwrap()).await.expect("the app answers");
    let allowed = response.headers().get("access-control-allow-origin").map(|v| v.to_str().unwrap().to_string());
    (response.status(), allowed)
}

const PLATFORM: &str = "https://api.smartloop.ai";

#[tokio::test]
async fn a_host_that_isnt_this_machine_is_refused() {
    for host in ["127.0.0.1:38540", "localhost:5000", "[::1]:80", "app.localhost"] {
        assert_eq!(send(Method::GET, "/", &[("host", host)]).await.0, StatusCode::OK, "{host}");
    }
    // A site that points its own name at 127.0.0.1.
    for host in ["evil.example:38540", "192.168.1.20:38540"] {
        assert_eq!(send(Method::GET, "/", &[("host", host)]).await.0, StatusCode::FORBIDDEN, "{host}");
    }
}

#[tokio::test]
async fn a_page_of_another_site_is_refused() {
    let host = ("host", "127.0.0.1:38540");
    // A client that isn't a browser sends no Origin.
    assert_eq!(send(Method::GET, "/", &[host]).await.0, StatusCode::OK);
    // A page of this machine's, as the desktop app's renderer in development.
    let (status, allowed) = send(Method::GET, "/", &[host, ("origin", "http://localhost:5000")]).await;
    assert_eq!((status, allowed.as_deref()), (StatusCode::OK, Some("http://localhost:5000")));
    // Any other site, preflight or not.
    let evil = ("origin", "https://evil.example");
    assert_eq!(send(Method::GET, "/v1/projects", &[host, evil]).await.0, StatusCode::FORBIDDEN);
    let preflight = [host, evil, ("access-control-request-method", "POST")];
    assert_eq!(send(Method::OPTIONS, "/v1/projects/p/mcp/register", &preflight).await.0, StatusCode::FORBIDDEN);
    assert_eq!(send(Method::POST, "/v1/projects/p/mcp/register", &[host, evil]).await.0, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn the_platform_may_hand_over_a_sign_in_and_nothing_else() {
    let host = ("host", "127.0.0.1:38540");
    let platform = ("origin", PLATFORM);
    let preflight = |method| [host, platform, ("access-control-request-method", method)];
    let (status, allowed) = send(Method::OPTIONS, "/v1/auth/callback", &preflight("POST")).await;
    assert_eq!((status, allowed.as_deref()), (StatusCode::OK, Some(PLATFORM)));
    assert_eq!(send(Method::OPTIONS, "/v1/auth/callback", &preflight("DELETE")).await.0, StatusCode::FORBIDDEN);
    assert_eq!(send(Method::OPTIONS, "/v1/projects", &preflight("POST")).await.0, StatusCode::FORBIDDEN);
    assert_eq!(send(Method::GET, "/v1/auth/status", &[host, platform]).await.0, StatusCode::FORBIDDEN);
    // The web app's origin isn't the platform's.
    let web = ("origin", "https://app.smartloop.ai");
    let preflight = [host, web, ("access-control-request-method", "POST")];
    assert_eq!(send(Method::OPTIONS, "/v1/auth/callback", &preflight).await.0, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn another_sites_request_without_an_origin_is_refused_but_the_sign_in_link() {
    let host = ("host", "127.0.0.1:38540");
    let cross = ("sec-fetch-site", "cross-site");
    // An <img> or a link from another site.
    assert_eq!(send(Method::GET, "/v1/projects", &[host, cross, ("sec-fetch-mode", "no-cors")]).await.0, StatusCode::FORBIDDEN);
    assert_eq!(send(Method::GET, "/", &[host, cross, ("sec-fetch-mode", "navigate")]).await.0, StatusCode::FORBIDDEN);
    // The sign-in hand-off's fallback, followed as a link, reaches its route.
    let followed = send(Method::GET, "/v1/auth/callback", &[host, cross, ("sec-fetch-mode", "navigate")]).await.0;
    assert_ne!(followed, StatusCode::FORBIDDEN);
    // The desktop app's renderer: cors mode, no Origin.
    assert_eq!(send(Method::GET, "/", &[host, cross, ("sec-fetch-mode", "cors")]).await.0, StatusCode::OK);
    // Typed into the address bar.
    assert_eq!(send(Method::GET, "/", &[host, ("sec-fetch-site", "none"), ("sec-fetch-mode", "navigate")]).await.0, StatusCode::OK);
}
