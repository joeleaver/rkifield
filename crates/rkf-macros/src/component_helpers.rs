//! Helper functions for the `#[component]` proc macro.
//!
//! Field type classification, get/set code generation, and attribute inspection.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, Token};

/// Check if a type is `[f32; 4]` (color array).
pub fn is_f32_array_4(ty: &syn::Type) -> bool {
    if let syn::Type::Array(arr) = ty {
        if let syn::Type::Path(p) = &*arr.elem {
            if p.path.segments.last().map(|s| s.ident == "f32").unwrap_or(false) {
                if let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(lit_int), .. }) = &arr.len {
                    return lit_int.base10_digits() == "4";
                }
            }
        }
    }
    false
}

/// Classify a type path segment to a FieldType variant name.
pub fn classify_type(ty: &syn::Type) -> &'static str {
    if is_f32_array_4(ty) {
        return "Color";
    }
    match ty {
        syn::Type::Path(type_path) => {
            let segments = &type_path.path.segments;
            let last = match segments.last() {
                Some(s) => s.ident.to_string(),
                None => return "String", // fallback
            };
            match last.as_str() {
                "f32" | "f64" => "Float",
                "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize"
                | "isize" => "Int",
                "bool" => "Bool",
                "Vec3" => "Vec3",
                "WorldPosition" => "WorldPosition",
                "Quat" => "Quat",
                "String" => "String",
                "Entity" => "Entity",
                "Option" => {
                    if let syn::PathArguments::AngleBracketed(args) = &segments.last().unwrap().arguments {
                        if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                            let inner_class = classify_type(inner_ty);
                            if inner_class == "Entity" {
                                return "Entity";
                            }
                        }
                    }
                    "String" // fallback for Option<Other>
                }
                "Vec" => "List",
                _ => "String", // fallback
            }
        }
        _ => "String",
    }
}

/// Check if a field has `#[serde(skip)]` among its attributes.
pub fn has_serde_skip(attrs: &[syn::Attribute]) -> bool {
    for attr in attrs {
        if attr.path().is_ident("serde") {
            if let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated,
            ) {
                for meta in &nested {
                    if meta.path().is_ident("skip") {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if a field has `#[persist]` among its attributes.
pub fn has_persist(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("persist"))
}

/// Check if a type is `Option<Entity>` (or `Option<hecs::Entity>`).
pub fn is_option_entity(ty: &syn::Type) -> bool {
    if let syn::Type::Path(type_path) = ty {
        if let Some(seg) = type_path.path.segments.last() {
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return classify_type(inner_ty) == "Entity";
                    }
                }
            }
        }
    }
    false
}

/// Get the last segment ident string from a type.
pub fn type_last_ident(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(type_path) => {
            type_path.path.segments.last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default()
        }
        _ => String::new(),
    }
}

/// Generate the get_field match arm for a given field.
pub fn gen_get_field_arm(field_name: &str, field_ident: &Ident, ty: &syn::Type, _struct_name: &Ident) -> Option<TokenStream2> {
    if is_f32_array_4(ty) {
        let expr = quote! { rkf_runtime::behavior::GameValue::Color(c.#field_ident) };
        return Some(quote! { #field_name => Ok(#expr), });
    }
    let last = type_last_ident(ty);
    if last == "Entity" || (last == "Option" && is_option_entity(ty)) {
        if is_option_entity(ty) {
            return Some(quote! {
                #field_name => {
                    let uuid_str = match c.#field_ident {
                        Some(ref_entity) => {
                            world.get::<&rkf_runtime::behavior::StableId>(ref_entity)
                                .map(|sid| sid.uuid().to_string())
                                .unwrap_or_default()
                        }
                        None => String::new(),
                    };
                    Ok(rkf_runtime::behavior::GameValue::String(uuid_str))
                },
            });
        } else {
            return Some(quote! {
                #field_name => {
                    let uuid_str = world.get::<&rkf_runtime::behavior::StableId>(c.#field_ident)
                        .map(|sid| sid.uuid().to_string())
                        .unwrap_or_default();
                    Ok(rkf_runtime::behavior::GameValue::String(uuid_str))
                },
            });
        }
    }
    let expr = match last.as_str() {
        "f32" => quote! { rkf_runtime::behavior::GameValue::Float(c.#field_ident as f64) },
        "f64" => quote! { rkf_runtime::behavior::GameValue::Float(c.#field_ident) },
        "i8" | "i16" | "i32" => quote! { rkf_runtime::behavior::GameValue::Int(c.#field_ident as i64) },
        "i64" => quote! { rkf_runtime::behavior::GameValue::Int(c.#field_ident) },
        "u8" | "u16" | "u32" => quote! { rkf_runtime::behavior::GameValue::Int(c.#field_ident as i64) },
        "u64" | "usize" | "isize" => quote! { rkf_runtime::behavior::GameValue::Int(c.#field_ident as i64) },
        "bool" => quote! { rkf_runtime::behavior::GameValue::Bool(c.#field_ident) },
        "String" => quote! { rkf_runtime::behavior::GameValue::String(c.#field_ident.clone()) },
        "Vec3" => quote! { rkf_runtime::behavior::GameValue::Vec3(c.#field_ident) },
        "Quat" => quote! { rkf_runtime::behavior::GameValue::Quat(c.#field_ident) },
        "WorldPosition" => quote! { rkf_runtime::behavior::GameValue::WorldPosition(c.#field_ident.clone()) },
        _ => return None,
    };
    Some(quote! { #field_name => Ok(#expr), })
}

/// Generate the set_field match arm for a given field.
pub fn gen_set_field_arm(field_name: &str, field_ident: &Ident, ty: &syn::Type, struct_name: &Ident) -> Option<TokenStream2> {
    let struct_name_str = struct_name.to_string();
    if is_f32_array_4(ty) {
        let expr = quote! {
            rkf_runtime::behavior::GameValue::Color(col) => { c.#field_ident = col; }
        };
        return Some(quote! { #field_name => match value { #expr _ => return Err(format!("type mismatch for field '{}'", field_name)), }, });
    }
    let last = type_last_ident(ty);
    if last == "Entity" || (last == "Option" && is_option_entity(ty)) {
        if is_option_entity(ty) {
            return Some(quote! {
                #field_name => {
                    drop(c);
                    let resolved = match value {
                        rkf_runtime::behavior::GameValue::String(uuid_str) => {
                            if uuid_str.is_empty() {
                                Ok(None)
                            } else {
                                let target_uuid = rkf_runtime::behavior::_MacroUuid::parse_str(&uuid_str)
                                    .map_err(|e| format!("invalid UUID for field '{}': {}", field_name, e))?;
                                let mut found = None;
                                for (ent, sid) in world.query::<&rkf_runtime::behavior::StableId>().iter() {
                                    if sid.uuid() == target_uuid {
                                        found = Some(ent);
                                        break;
                                    }
                                }
                                Ok(Some(found.ok_or_else(|| {
                                    format!("no entity with StableId '{}' found for field '{}'", uuid_str, field_name)
                                })?))
                            }
                        }
                        _ => Err(format!("type mismatch for field '{}'", field_name)),
                    }?;
                    let mut c = world.get::<&mut #struct_name>(entity)
                        .map_err(|_| format!("entity does not have component '{}'", #struct_name_str))?;
                    c.#field_ident = resolved;
                },
            });
        } else {
            return Some(quote! {
                #field_name => {
                    drop(c);
                    let resolved = match value {
                        rkf_runtime::behavior::GameValue::String(uuid_str) => {
                            let target_uuid = rkf_runtime::behavior::_MacroUuid::parse_str(&uuid_str)
                                .map_err(|e| format!("invalid UUID for field '{}': {}", field_name, e))?;
                            let mut found = None;
                            for (ent, sid) in world.query::<&rkf_runtime::behavior::StableId>().iter() {
                                if sid.uuid() == target_uuid {
                                    found = Some(ent);
                                    break;
                                }
                            }
                            found.ok_or_else(|| {
                                format!("no entity with StableId '{}' found for field '{}'", uuid_str, field_name)
                            })
                        }
                        _ => Err(format!("type mismatch for field '{}'", field_name)),
                    }?;
                    let mut c = world.get::<&mut #struct_name>(entity)
                        .map_err(|_| format!("entity does not have component '{}'", #struct_name_str))?;
                    c.#field_ident = resolved;
                },
            });
        }
    }
    let expr = match last.as_str() {
        "f32" => quote! { rkf_runtime::behavior::GameValue::Float(f) => { c.#field_ident = f as f32; } },
        "f64" => quote! { rkf_runtime::behavior::GameValue::Float(f) => { c.#field_ident = f; } },
        "i8" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as i8; } },
        "i16" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as i16; } },
        "i32" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as i32; } },
        "i64" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i; } },
        "u8" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as u8; } },
        "u16" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as u16; } },
        "u32" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as u32; } },
        "u64" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as u64; } },
        "usize" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as usize; } },
        "isize" => quote! { rkf_runtime::behavior::GameValue::Int(i) => { c.#field_ident = i as isize; } },
        "bool" => quote! { rkf_runtime::behavior::GameValue::Bool(b) => { c.#field_ident = b; } },
        "String" => quote! { rkf_runtime::behavior::GameValue::String(s) => { c.#field_ident = s; } },
        "Vec3" => quote! { rkf_runtime::behavior::GameValue::Vec3(v) => { c.#field_ident = v; } },
        "Quat" => quote! { rkf_runtime::behavior::GameValue::Quat(q) => { c.#field_ident = q; } },
        "WorldPosition" => quote! { rkf_runtime::behavior::GameValue::WorldPosition(wp) => { c.#field_ident = wp; } },
        _ => return None,
    };
    Some(quote! { #field_name => match value { #expr _ => return Err(format!("type mismatch for field '{}'", field_name)), }, })
}
